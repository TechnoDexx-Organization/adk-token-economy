"""
cache_probe.py — замер кэширования промпта у Yandex AI Studio и Selectel AI Gateway.

Пишет СЫРОЙ лог (полный `usage` каждого ответа) в JSONL-файл рядом со скриптом,
чтобы выводы перепроверялись по источнику, а не по распечатке.

Что проверяет:
  1. Факт кэширования: появляется ли `cached_tokens > 0` на повторном запросе.
  2. Влияние хедера `OpenAI-Project` (его подставляет LiteLLM) на кэш.
  3. Порог кэширования — длиной префикса (`--repeat`).
  4. TTL кэша — паузой между запросами (`--pause`).

Примеры:
    python cache_probe.py                      # базовый прогон (префикс ~1182, пауза 1с)
    python cache_probe.py --pause 120          # TTL: жив ли кэш через 2 минуты
    python cache_probe.py --repeat 43          # порог: префикс ~512 токена
    python cache_probe.py --only yandex        # только Яндекс (Selectel на 402)
"""


import argparse
import contextlib
import json
import os
import time
from datetime import datetime, timezone

import requests

with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SELECTEL_BASE = "https://api.selectel.ru/ai/v1"

UNIT_SENTENCE = "Это предложение повторяется много раз для создания объема. "
SYSTEM_HEAD = (
    "Ты — полезный ассистент. Твоя задача — повторять текст пользователя. "
    "Вот очень длинная инструкция, чтобы заполнить контекст и вызвать кэширование. "
)

# Замерено в первом прогоне: --repeat 100 даёт префикс ~1182 токена (cached=1024).
# Для теста порога ~512 токена используйте --repeat 43.
REF_REPEAT, REF_TOKENS = 100, 1182

# Соль: случайная строка в начало промпта, чтобы прошлый кэш не мешал.
# Без соли — повторный прогон с тем же префиксом может хитнуть старый кэш,
# и «cached=0» будет означать «не тот кэш», а не «ниже порога».
FRESH_SALT = "Xk9mQ2vL7pR4"


def build_prompt(repeat: int, fresh: bool = False) -> str:
    """Собирает системный промпт заданной длины.

    fresh=True — вставляет случайную соль в начало, чтобы промпт не хитнул
    кэш из прошлого прогона (чистый TTL-тест).
    """
    salt = f"{FRESH_SALT}{int(time.time())} " if fresh else ""
    return salt + SYSTEM_HEAD + UNIT_SENTENCE * repeat


def estimate_tokens(repeat: int) -> int:
    """Грубая оценка длины префикса по измеренной точке."""
    return round(REF_TOKENS * repeat / REF_REPEAT)


def extract_usage(data: dict) -> dict:
    """Достаёт счётчики токенов из OpenAI-совместимого ответа."""
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": details.get("cached_tokens", usage.get("cached_tokens")),
        "raw_usage": usage,
    }


def send_request(provider: str, prompt: str, extra_headers: dict | None = None) -> dict:
    """Один запрос к провайдеру. Никогда не бросает исключение — всё в результат."""
    yandex_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    yandex_base = os.getenv("YANDEX_API_BASE", "https://ai.api.cloud.yandex.net/v1")
    selectel_key = os.getenv("SELECTEL_API_KEY")

    if provider == "yandex":
        if not (yandex_key and folder_id):
            return {"status": "SKIP", "error": "YANDEX_API_KEY / YANDEX_FOLDER_ID не заданы"}
        url = f"{yandex_base}/chat/completions"
        headers = {
            "Authorization": f"Api-Key {yandex_key}",
            "Content-Type": "application/json",
        }
        model = f"gpt://{folder_id}/deepseek-v4.1-flash/latest"
    elif provider == "selectel":
        if not selectel_key:
            return {"status": "SKIP", "error": "SELECTEL_API_KEY не задан"}
        url = f"{SELECTEL_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {selectel_key}",
            "Content-Type": "application/json",
        }
        model = "deepseek/deepseek-v4.1-flash"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Привет! Повтори это сообщение."},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        return {"status": "NETWORK_ERROR", "error": str(e), "model": model}

    record = {
        "status": "OK" if resp.ok else "HTTP_ERROR",
        "http_status": resp.status_code,
        "model": model,
    }

    if not resp.ok:
        record["error_body"] = resp.text[:500]
        return record

    try:
        record |= extract_usage(resp.json())
    except ValueError:
        record["status"] = "NOT_JSON"
        record["error_body"] = resp.text[:500]

    return record


def probe_pair(
    provider: str,
    prompt: str,
    pause: float,
    extra_headers: dict | None,
    log_fh,
    scenario: str,
) -> tuple[dict, dict]:
    """Два идентичных запроса с паузой. Пишет обе записи в JSONL, возвращает их."""
    first = send_request(provider, prompt, extra_headers)
    log_fh.write(
        json.dumps(
            {"scenario": scenario, "provider": provider, "attempt": 1,
             "ts": datetime.now(timezone.utc).isoformat(), **first},
            ensure_ascii=False,
        ) + "\n"
    )
    log_fh.flush()

    time.sleep(pause)

    second = send_request(provider, prompt, extra_headers)
    log_fh.write(
        json.dumps(
            {"scenario": scenario, "provider": provider, "attempt": 2,
             "ts": datetime.now(timezone.utc).isoformat(), **second},
            ensure_ascii=False,
        ) + "\n"
    )
    log_fh.flush()
    return first, second


def fmt(record: dict) -> str:
    if record.get("status") != "OK":
        detail = record.get("error") or record.get("error_body", "")
        return f"{record.get('status')} ({record.get('http_status', '-')}) {detail[:60]}"
    return (
        f"prompt={record.get('prompt_tokens')}, "
        f"cached={record.get('cached_tokens')}"
    )


def cached_of(record: dict) -> int:
    value = record.get("cached_tokens")
    return value if isinstance(value, int) else 0


def run_scenario(
    label: str,
    key: str,
    provider: str,
    prompt: str,
    pause: float,
    extra_headers: dict | None,
    log_fh,
) -> tuple[dict, dict]:
    """Прогон одного сценария: заголовок, пара запросов, печать результатов."""
    print(label)
    pair = probe_pair(provider, prompt, pause, extra_headers, log_fh, key)
    print(f"   1-й: {fmt(pair[0])}")
    print(f"   2-й: {fmt(pair[1])}\n")
    return pair


def main() -> None:  # sourcery skip: low-code-quality
    parser = argparse.ArgumentParser(description="Замер кэширования промпта")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="пауза между запросами пары, сек (для TTL-теста)")
    parser.add_argument("--repeat", type=int, default=REF_REPEAT,
                        help=f"длина префикса; {REF_REPEAT} ≈ {REF_TOKENS} токенов")
    parser.add_argument("--only", choices=["yandex", "selectel"], default=None)
    parser.add_argument("--out", default=None, help="путь к JSONL-логу")
    parser.add_argument("--fresh", action="store_true",
                        help="добавить случайную соль в начало промпта (чистый TTL-тест)")
    args = parser.parse_args()

    prompt = build_prompt(args.repeat, fresh=args.fresh)
    out_path = args.out or os.path.join(
        BASE_DIR,
        f"cache_probe_{datetime.now():%Y%m%d_%H%M%S}.jsonl",  # noqa: DTZ005
    )

    print("=== ТЕСТ КЭШИРОВАНИЯ ===")
    print(f"префикс: ~{estimate_tokens(args.repeat)} токенов (--repeat {args.repeat})")
    print(f"пауза между запросами: {args.pause} сек")
    print(f"сырой лог: {out_path}\n")

    results: dict[str, tuple[dict, dict]] = {}

    with open(out_path, "a", encoding="utf-8") as log_fh:
        if args.only in (None, "yandex"):
            results["yandex_clean"] = run_scenario(
                "1. Yandex, чистый запрос", "yandex_clean",
                "yandex", prompt, args.pause, None, log_fh,
            )
            results["yandex_header"] = run_scenario(
                "2. Yandex + OpenAI-Project (как подставляет LiteLLM)", "yandex_header",
                "yandex", prompt, args.pause,
                {"OpenAI-Project": os.getenv("YANDEX_FOLDER_ID", "test")}, log_fh,
            )

        if args.only in (None, "selectel"):
            results["selectel_clean"] = run_scenario(
                "3. Selectel, чистый запрос", "selectel_clean",
                "selectel", prompt, args.pause, None, log_fh,
            )

    print("=== ВЫВОДЫ ===")
    if args.only in (None, "yandex"):
        clean_1, clean_2 = results["yandex_clean"]
        head_1, _head_2 = results["yandex_header"]

        if clean_2.get("status") != "OK":
            print("⚠️ Яндекс: запрос не выполнен, выводы делать нельзя.")
        elif cached_of(clean_2) > 0:
            print(f"✅ Яндекс кэширует: {cached_of(clean_2)} из "
                  f"{clean_2.get('prompt_tokens')} токенов.")
        else:
            print("❌ Яндекс: на повторном запросе cached=0.")

        if cached_of(clean_2) > 0 and cached_of(head_1) > 0:
            print("✅ Хедер OpenAI-Project кэш НЕ ломает "
                  "(первый запрос с хедером попал в уже прогретый кэш).")
        elif cached_of(clean_2) > 0 and cached_of(head_1) == 0:
            print("⚠️ С хедером кэш не сработал — гипотеза про LiteLLM-хедер жива.")

        if cached_of(clean_1) > 0:
            print("⚠️ Первый запрос уже кэшированный — кэш пережил паузу/прошлые прогоны.")

    if args.only in (None, "selectel"):
        _sel_1, sel_2 = results["selectel_clean"]
        if sel_2.get("status") != "OK":
            print(f"❌ Selectel недоступен: {sel_2.get('status')} "
                  f"{sel_2.get('error_body', '')[:60]}")
        elif cached_of(sel_2) > 0:
            print("✅ Selectel кэширует.")
        else:
            print("❌ Selectel: на повторном запросе cached=0.")

    if args.pause > 10:
        print("ℹ️ Пауза > 10 сек — это TTL-тест: смотри, остался ли cached на 2-м запросе.")
    if estimate_tokens(args.repeat) < 1024:
        print("ℹ️ Префикс < 1024 токенов — это тест порога: ждём cached=0 при коротком префиксе.")


if __name__ == "__main__":
    main()
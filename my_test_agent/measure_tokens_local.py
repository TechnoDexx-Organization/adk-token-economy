import tiktoken
from transformers import AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-V2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer is None:
    raise RuntimeError(
        f"Не удалось загрузить токенизатор {MODEL_ID}. "
        "Проверь: pip install sentencepiece, "
        "pip install --upgrade transformers, и что ID модели верный."
    )

texts = {
    "обычный": "Я устал, потому что долго работал, и теперь не могу сосредоточиться",
    "разреженный": "Устал. Долго работал. Не могу сосредоточиться",
    "предикаты": "устал(я). работал(долго). сосредоточиться(не_могу).",
}

for name, text in texts.items():
    tokens = tokenizer.encode(text)
    print(f"{name:15} → {len(tokens):4} токенов")
    
print('\n==TikToken==\n')
enc = tiktoken.get_encoding("cl100k_base")

for name, text in texts.items():
    print(f"{name:15} → {len(enc.encode(text)):4} токенов")
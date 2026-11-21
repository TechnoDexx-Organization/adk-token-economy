"""Защита от SSRF (Server-Side Request Forgery).

`HttpMemoryProvider` ходит во внешний RAG по URL, который приходит из
конфигурации. Если этот URL указывает на `127.0.0.1`, `localhost` или
приватную подсеть, агент через память агента получает доступ к внутренней
инфраструктуре хоста (метаданные облака, админки, БД). Модуль даёт чистую
функцию проверки, которую можно переиспользовать в HTTP-инструментах и
guard-плагинах.
"""

import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# localhost и метаданные облачных провайдеров, которые резолвятся не всегда
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}

# Адреса, которые нельзя посещать даже в явном виде (IPv4 + IPv6)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",  # «эта» сеть + 0.0.0.0
        "10.0.0.0/8",  # RFC1918
        "100.64.0.0/10",  # CGNAT, у Yandex Cloud metadata на 169.254
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local / cloud metadata
        "172.16.0.0/12",  # RFC1918
        "192.168.0.0/16",  # RFC1918
        "::1/128",  # IPv6 loopback
        "fc00::/7",  # IPv6 unique-local
        "fe80::/10",  # IPv6 link-local
    )
]

# IPv4, «упакованный» в IPv6-префикс (например, ::ffff:127.0.0.1)
_V4_MAPPED_PREFIX = ipaddress.ip_network("::ffff:0:0/96")


def _is_blocked_ip(raw: str) -> bool:
    """True, если адрес попадает в запрещённую сеть."""
    try:
        ip = ipaddress.ip_address(raw.strip("[]"))
    except ValueError:
        # Это не IP — хостнеймы проверяет вызывающий код
        return False
    if ip.version == 6:
        mapped = ip.ipv4_mapped
        if mapped is not None:
            ip = mapped
    return any(ip in network for network in _BLOCKED_NETWORKS)


def is_safe_url(url: str) -> bool:
    """
    Разрешён ли HTTP-запрос к этому URL.

    Запрещены: не-HTTP схемы, пустой хост, localhost-алиасы, прямые IP из
    приватных/служебных сетей и хостнеймы, которые резолвятся в такие адреса.
    Ошибка резолва — тоже отказ (fail closed).
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    try:
        host = parsed.hostname
    except ValueError:
        return False
    if not host:
        return False

    host = host.rstrip(".").lower()
    if host in _BLOCKED_HOSTNAMES or host.endswith(".localhost") or host.endswith(".local"):
        return False

    if _is_blocked_ip(host):
        return False

    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        logger.warning(
            "SSRF: не удалось разрешить хост %s — запрос запрещён", host, extra={"kind": "ssrf_dns_fail", "host": host}
        )
        return False
    for info in infos:
        if info[0] not in (socket.AF_INET, socket.AF_INET6):
            continue
        addr_raw = info[4][0]
        addr = addr_raw if isinstance(addr_raw, str) else str(addr_raw)
        if _is_blocked_ip(addr):
            return False
    return True


# URL в произвольном тексте (для проверки аргументов инструментов)
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s'\"<>)]+", re.IGNORECASE)


def find_blocked_urls(text: str) -> list[str]:
    """Возвращает все SSRF-адреса, упомянутые в тексте (для audit-логов)."""
    if not text:
        return []
    return [url for url in _URL_IN_TEXT_RE.findall(text) if not is_safe_url(url)]

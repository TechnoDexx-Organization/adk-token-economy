import logging
import re
import shlex

from ..core.interfaces import GuardDecision, ToolGuard

logger = logging.getLogger(__name__)


class SafeToolGuard(ToolGuard):
    """Простой Guard по спискам разрешенных/запрещенных инструментов."""
    def __init__(self, allow_list: list[str] | None = None, deny_list: list[str] | None = None):
        self.allow_list = allow_list or []
        self.deny_list = deny_list or []

    def check(self, tool_name: str, args: dict, context: dict) -> GuardDecision:
        if tool_name in self.deny_list:
            logger.warning("🚫 Блокировка (deny_list): %s", tool_name)
            return GuardDecision.DENY
        if self.allow_list and tool_name not in self.allow_list:
            logger.warning("🚫 Блокировка (не в allow_list): %s", tool_name)
            return GuardDecision.DENY
        return GuardDecision.ALLOW


class ShellCommandGuard(ToolGuard):
    """
    Гибридный Guard: DENY -> ALLOW -> HITL.
    """
    def __init__(self, allowed_commands: list[str] | None = None, deny_patterns: list[str] | None = None):
        self.deny_patterns = deny_patterns or [
            r"rm\s+-[rRfF]+",          # Удаление с рекурсией/силой
            r"mkfs",                   # Форматирование
            r"dd\s+if=",               # Запись на диск
            r":\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
            r"sudo",                   # Повышение привилегий
            r">\s*/dev/sd[a-z]",       # Запись напрямую в диск
            r"chmod\s+-R\s+777",       # Раздача прав всем
        ]
        
        default_allowed = [
            "ls", "cat", "grep", "echo", "pwd", "git", "pytest", 
            "python", "uv", "find", "head", "tail", "wc", "diff"
        ]
        self.allowed = set(allowed_commands or default_allowed)

    def check(self, tool_name: str, args: dict, context: dict) -> GuardDecision:
        if tool_name not in ["run_shell_command", "execute_command"]:
            return GuardDecision.ALLOW

        command = args.get("command", "").strip()
        if not command:
            return GuardDecision.DENY

        # ШАГ 1: DENY
        for pattern in self.deny_patterns:
            if re.search(pattern, command):
                logger.warning("🚫 Guard (DENY): Заблокирована опасная команда: %s", command)
                return GuardDecision.DENY

        # ШАГ 2: Разбор на токены
        try:
            tokens = shlex.split(command)
            if not tokens:
                return GuardDecision.DENY
            first_token = tokens[0]
        except ValueError:
            logger.warning("🚫 Guard: Ошибка парсинга команды: %s", command)
            return GuardDecision.DENY

        # ШАГ 3: ALLOW
        if first_token in self.allowed:
            return GuardDecision.ALLOW

        # ШАГ 4: HITL (Серая зона)
        logger.warning("⚠️ Guard (HITL): Неизвестная команда '%s'. Требуется подтверждение.", first_token)
        return GuardDecision.REQUIRE_HITL
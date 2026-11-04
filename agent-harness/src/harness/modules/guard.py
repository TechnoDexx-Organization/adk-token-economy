import re

from ..core.interfaces import ToolGuard


class SafeToolGuard(ToolGuard):
    """Простой Guard по спискам разрешенных/запрещенных инструментов."""
    def __init__(self, allow_list: list[str] | None = None, deny_list: list[str] | None = None):
        self.allow_list = allow_list or []
        self.deny_list = deny_list or []

    def check(self, tool_name: str, args: dict, context: dict) -> bool:
        if tool_name in self.deny_list:
            print(f"🚫 Блокировка (deny_list): {tool_name}")
            return False
        if self.allow_list and tool_name not in self.allow_list:
            print(f"🚫 Блокировка (не в allow_list): {tool_name}")
            return False
        return True


class ShellCommandGuard(ToolGuard):
    """
    Guard для shell-команд. Блокирует опасные паттерны (rm -rf, mkfs и т.д.),
    но пропускает безопасные команды (ls, cat, git status).
    """
    def __init__(self, deny_patterns: list[str] | None = None):
        # Список регулярных выражений для запрещенных команд
        self.deny_patterns = deny_patterns or [
            r"rm\s+-rf",          # Удаление с рекурсией и без подтверждения
            r"mkfs",              # Форматирование диска
            r"dd\s+if=",          # Запись на диск
            r":\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
            r"chmod\s+-R\s+777",  # Раздача прав всем
            r"curl.*\|\s*bash",   # Pipe to bash (инъекция)
            r"wget.*\|\s*bash",
            r">\s*/dev/sd[a-z]",  # Запись напрямую в диск
        ]

    def check(self, tool_name: str, args: dict, context: dict) -> bool:
        # Нас интересуют только вызовы shell-инструментов
        if tool_name not in ["run_shell_command", "execute_command"]:
            return True

        command = args.get("command", "")
        for pattern in self.deny_patterns:
            if re.search(pattern, command):
                print(f"🚫 Блокировка опасной команды: {command}")
                return False
        
        return True
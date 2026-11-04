from ..core.interfaces import ToolGuard

class SafeToolGuard(ToolGuard):
    def __init__(self, allow_list: list = None, deny_list: list = None):
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

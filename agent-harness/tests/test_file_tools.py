
from harness.modules.file_tools import FileTools


def test_run_shell_command_allows_safe_command(tmp_path):
    """Проверяет, что безопасная команда выполняется."""
    tools = FileTools(base_dir=str(tmp_path))
    result = tools.run_shell_command("echo 'hello world'")
    
    assert "hello world" in result
    assert "Ошибка безопасности" not in result


def test_run_shell_command_blocks_dangerous_command(tmp_path):
    """Проверяет, что ShellCommandGuard блокирует опасную команду внутри FileTools."""
    tools = FileTools(base_dir=str(tmp_path))
    # Пытаемся выполнить команду, которая есть в deny_patterns у ShellCommandGuard
    result = tools.run_shell_command("chmod -R 777 /")
    
    # Убеждаемся, что команда была заблокирована и вернула наше сообщение
    assert "Ошибка безопасности" in result
    assert "Команда заблокирована" in result


def test_read_file_size_limit(tmp_path):
    """Проверяет, что файлы больше 50 КБ не читаются."""
    tools = FileTools(base_dir=str(tmp_path))
    big_file = tmp_path / "big.txt"
    # Создаем файл размером чуть больше 50 000 байт
    big_file.write_text("A" * 50_001, encoding="utf-8")
    
    result = tools.read_file("big.txt")
    assert "файл слишком большой" in result
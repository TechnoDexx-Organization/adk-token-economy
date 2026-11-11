import os
import subprocess
from pathlib import Path


class FileTools:
    """Инструменты для безопасной работы с файловой системой и запуска команд."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()

    def run_shell_command(self, command: str) -> str:
        """Выполняет shell-команду с таймаутом и в рабочей директории проекта."""
        try:
            result = subprocess.run(
                command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=self.base_dir,
            )
            output = result.stdout.strip()
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr.strip()}"
            return output or "Команда выполнена успешно (нет вывода)."
        except subprocess.TimeoutExpired:
            return "❌ Ошибка: Команда превысила лимит времени (15 секунд)."
        except Exception as e:  # noqa: BLE001
            return f"❌ Ошибка выполнения: {e!s}"

    def write_file(self, filename: str, content: str) -> str:
        """Записывает код в файл внутри проекта."""
        file_path = self._resolve_safe_path(filename)
        if isinstance(file_path, str):  # Ошибка безопасности
            return file_path

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ Файл '{filename}' успешно записан."
        except Exception as e:  # noqa: BLE001
            return f"❌ Ошибка записи файла: {e!s}"

    def read_file(self, filename: str) -> str:
        """Читает содержимое файла внутри проекта (макс 50 КБ)."""
        file_path = self._resolve_safe_path(filename)
        if isinstance(file_path, str):  # Ошибка безопасности
            return file_path

        if not file_path.exists():
            return f"❌ Файл '{filename}' не найден."

        try:
            if file_path.stat().st_size > 50_000:
                return "❌ Ошибка: файл слишком большой (больше 50 КБ)."
            return file_path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return f"❌ Ошибка чтения файла: {e!s}"

    def _resolve_safe_path(self, filename: str) -> Path | str:
        """Проверяет, что путь не выходит за пределы base_dir."""
        if ".." in filename or filename.startswith("/"):
            return "❌ Ошибка безопасности: выход за пределы проекта запрещен."
        
        target = (self.base_dir / filename).resolve()
        if not target.is_relative_to(self.base_dir):
            return "❌ Ошибка безопасности: выход за пределы проекта запрещен."
        
        return target

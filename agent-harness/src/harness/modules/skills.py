import pathlib

from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

# Импортируем твой класс из соседнего файла
from .guard import ShellCommandGuard


def build_safe_skill_toolset(skills_dir_path: pathlib.Path) -> skill_toolset.SkillToolset:
    loaded_skills = []

    # Инициализируем твой охранник
    guard = ShellCommandGuard()

    for skill_dir in skills_dir_path.iterdir():
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
            continue

        scripts_dir = skill_dir / "scripts"
        if scripts_dir.exists():
            for script in scripts_dir.iterdir():
                if script.is_file():
                    try:
                        # Читаем текст скрипта
                        content = script.read_text(encoding="utf-8")

                        # Хитрость: имитируем вызов инструмента shell, 
                        # чтобы твой ShellCommandGuard проверил этот скрипт!
                        is_safe = guard.check(
                            tool_name="run_shell_command", 
                            args={"command": content}, 
                            context={}
                        )

                        if not is_safe:
                            raise ValueError(
                                f"Загрузка навыка '{skill_dir.name}' отменена: "
                                f"опасный скрипт '{script.name}'."
                            )
                    except Exception as e:
                        # Если не смогли прочитать или сработал guard
                        raise ValueError(f"Ошибка при проверке {script.name}: {e}") from e

        # Если проверки пройдены, загружаем навык
        skill = load_skill_from_dir(skill_dir)
        loaded_skills.append(skill)

    return skill_toolset.SkillToolset(skills=loaded_skills)
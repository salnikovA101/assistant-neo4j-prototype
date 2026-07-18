import logging
from pathlib import Path

from server.utils.constants import TTSModes

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Класс для загрузки и управления текстовыми промптами.
    Загружает базовую логику (assistant_logic.md) и, при включённом аудио,
    формат вывода для TTS (output_{mode}.md).
    """

    def __init__(
        self, folder_name: str, mode: TTSModes, audio_enabled: bool = True
    ) -> None:
        """
        Инициализирует загрузчик и считывает файлы.
        """
        self.logic_text = ""
        self.output_text = ""
        self.mode = mode
        self.audio_enabled = audio_enabled
        self._load(folder_name)

    def _load(self, folder_name: str) -> None:
        path = Path(folder_name)
        if not path.is_dir():
            logger.error(
                f"Путь {folder_name} не существует или не является директорией."
            )
            return

        logic_file = path / "assistant_logic.md"
        output_file_speed = path / "output_speed.md"
        output_file_quality = path / "output_quality.md"

        try:
            if logic_file.exists():
                self.logic_text = logic_file.read_text(encoding="utf-8").strip()
                logger.info("Промпт assistant_logic.md успешно загружен.")
            else:
                logger.warning("Файл assistant_logic.md не найден.")

            if not self.audio_enabled:
                logger.info("Аудио выключено — промпт для озвучки не загружается.")
                return

            if self.mode == TTSModes.SPEED:
                if output_file_speed.exists():
                    self.output_text = output_file_speed.read_text(
                        encoding="utf-8"
                    ).strip()
                    logger.info("Промпт output_speed.md успешно загружен.")
                else:
                    logger.warning("Файл output_speed.md не найден.")
            elif self.mode == TTSModes.QUALITY or self.mode == TTSModes.CLOUD:
                if output_file_quality.exists():
                    self.output_text = output_file_quality.read_text(
                        encoding="utf-8"
                    ).strip()
                    logger.info("Промпт output_quality.md успешно загружен.")
                else:
                    logger.warning("Файл output_quality.md не найден.")

        except Exception as e:
            logger.error(f"Ошибка при чтении файлов промптов: {e}")

    def get_system_prompt(self) -> str:
        """
        Возвращает объединенный текст промптов.
        """
        if not self.output_text:
            return self.logic_text.strip()
        return f"{self.logic_text}\n\n{self.output_text}".strip()

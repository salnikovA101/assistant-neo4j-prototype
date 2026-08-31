import logging
from pathlib import Path

from server.utils.constants import TTSModes

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Класс для загрузки и управления текстовыми промптами.
    Загружает отдельную логику auto/staged/card и, при включённом аудио,
    формат вывода для TTS. TTS-формат никогда не добавляется к card prompt.
    """

    def __init__(
        self, folder_name: str, mode: TTSModes, audio_enabled: bool = True
    ) -> None:
        """
        Инициализирует загрузчик и считывает файлы.
        """
        self.logic_text = ""
        self.staged_text = ""
        self.card_text = ""
        self.output_text = ""
        self.mode = mode
        self.audio_enabled = audio_enabled
        self._load(folder_name)

    def _load(self, folder_name: str) -> None:
        path = Path(folder_name)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Путь {folder_name} не существует или не является директорией."
            )

        logic_file = path / "assistant_logic.md"
        staged_file = path / "assistant_staged.md"
        card_file = path / "card_generation.md"
        output_file_speed = path / "output_speed.md"
        output_file_quality = path / "output_quality.md"

        try:
            required = {
                "assistant_logic.md": logic_file,
                "assistant_staged.md": staged_file,
                "card_generation.md": card_file,
            }
            missing = [name for name, file in required.items() if not file.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Не найдены обязательные промпты: {', '.join(missing)}"
                )
            self.logic_text = logic_file.read_text(encoding="utf-8").strip()
            self.staged_text = staged_file.read_text(encoding="utf-8").strip()
            self.card_text = card_file.read_text(encoding="utf-8").strip()
            empty = [
                name
                for name, text in (
                    ("assistant_logic.md", self.logic_text),
                    ("assistant_staged.md", self.staged_text),
                    ("card_generation.md", self.card_text),
                )
                if not text
            ]
            if empty:
                raise ValueError(f"Пустые обязательные промпты: {', '.join(empty)}")
            logger.info("Промпты auto, staged и card успешно загружены.")

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

        except Exception:
            logger.exception("Ошибка при чтении файлов промптов")
            raise

    def get_system_prompt(self, mode: str = "auto") -> str:
        """
        Возвращает объединенный текст промптов.
        """
        if mode == "card":
            return self.card_text.strip()
        logic = self.staged_text if mode == "staged" else self.logic_text
        if not self.output_text:
            return logic.strip()
        return f"{logic}\n\n{self.output_text}".strip()

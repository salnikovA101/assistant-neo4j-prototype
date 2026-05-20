from typing import Any, AsyncGenerator

from server.utils.config import TtsConfig
from server.utils.constants import TTSModes
from server.tts.base import BaseTTSProvider
from server.tts.providers.cloud_tts import CloudTTSProvider
from server.tts.providers.quality_tts import QualityTTSProvider
from server.tts.providers.speed_tts import FastTTSProvider


class TTSManager:
    """
    Менеджер для управления провайдерами синтеза речи (TTS).

    Класс отвечает за инициализацию, динамическое переключение между
    различными режимами озвучки (скорость vs качество vs облако) и управление
    жизненным циклом моделей.
    """

    MODELS: dict[TTSModes, Any] = {
        TTSModes.SPEED: FastTTSProvider,
        TTSModes.QUALITY: QualityTTSProvider,
        TTSModes.CLOUD: CloudTTSProvider,
    }

    def __init__(self, config: TtsConfig) -> None:
        """
        Инициализирует менеджер TTS и загружает модель по умолчанию.

        Args:
            config (TtsConfig): Объект конфигурации, содержащий текущий режим
                и параметры для TTS-провайдеров.
        """
        self.config = config
        self.loaded_mode = config.mode
        self.model: BaseTTSProvider = self._load_model(self.config.mode)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь и yield'ит PCM-чанки через активную модель.

        Args:
            text (str): Текст для синтеза.

        Yields:
            bytes: PCM-чанки (int16, 24kHz, mono).
        """
        if self.loaded_mode != self.config.mode:
            self.unload()
            self.loaded_mode = self.config.mode
            self.model = self._load_model(self.config.mode)

        async for chunk in self.model.synthesize_stream(text):
            yield chunk

    def _load_model(self, mode: TTSModes) -> BaseTTSProvider:
        """
        Внутренний метод для создания экземпляра провайдера и его подготовки.

        Args:
            mode (TTSModes): Режим работы, определяющий выбор класса провайдера.

        Returns:
            BaseTTSProvider: Инициализированный и "прогретый" экземпляр провайдера.
        """
        model_class = self.MODELS.get(mode, FastTTSProvider)
        model = model_class(self.config)
        model.warmup()
        return model

    def unload(self) -> None:
        """
        Освобождает ресурсы текущей загруженной модели.
        """
        self.model.unload()

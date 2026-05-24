import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from server.utils.config import TtsConfig
from server.tts.base import BaseTTSProvider

logger = logging.getLogger(__name__)


class CloudTTSProvider(BaseTTSProvider):
    """
    Провайдер для облачных TTS совместимых с OpenAI API (например, OpenRouter).
    """

    def __init__(self, config: TtsConfig) -> None:
        self.config = config.cloud

        api_key = self.config.api_key
        base_url = self.config.base_url

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь из текста через облачный API и возвращает PCM-чанки (int16, 24kHz, mono).
        """
        if not text:
            return

        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.config.model,
                input=text,
                voice=self.config.voice,
                response_format="pcm",
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk
        except Exception as e:
            logger.warning(f"Облачный синтез речи прерван или недоступен: {e}")

    def unload(self) -> None:
        """
        Освобождает ресурсы. Для облачного API закрывает клиентскую сессию.
        """
        pass

    def warmup(self) -> None:
        """
        Выполняет предварительную подготовку (прогрев) модели.
        Для облака не требуется.
        """
        pass

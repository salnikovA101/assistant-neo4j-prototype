import asyncio
import logging
import time
import warnings
from typing import AsyncGenerator

import numpy as np
import torch

from server.utils.config import SpeedTtsConfig, TtsConfig
from server.tts.base import BaseTTSProvider

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


class FastTTSProvider(BaseTTSProvider):
    """
    Провайдер быстрого синтеза речи на базе Silero TTS.

    Обеспечивает минимальную задержку (latency) генерации.
    """

    def __init__(self, config: TtsConfig) -> None:
        """
        Инициализирует Silero TTS, загружая модель и перемещая её на целевое устройство.

        Args:
            config (TtsConfig): Общий объект конфигурации, из которого извлекаются
                настройки секции 'speed'.
        """
        self.config: SpeedTtsConfig = config.speed
        self.device = torch.device(self.config.device)
        logger.info("Загрузка Silero TTS V5...")
        self.model, _ = torch.hub.load(  # type: ignore
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.config.language,
            speaker=self.config.speaker_type,
        )
        self.model.to(self.device)
        logger.info(f"Silero TTS готов. Спикер: {self.config.speaker_name}")

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь и yield'ит PCM-чанки (int16).

        Silero генерирует аудио целиком, затем нарезаем на чанки по ~100мс.

        Args:
            text (str): Текст для синтеза.

        Yields:
            bytes: PCM-данные (int16, little-endian).
        """
        if not text:
            return

        start = time.perf_counter()
        audio = await asyncio.to_thread(
            self.model.apply_tts,
            text=text,
            speaker=self.config.speaker_name,
            sample_rate=self.config.sample_rate,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"TTS Time: {elapsed:.3f}s (Silero)")

        audio_np = audio.cpu().numpy()
        chunk_size = self.config.sample_rate // 10  # ~100мс чанки

        for i in range(0, len(audio_np), chunk_size):
            pcm = (audio_np[i : i + chunk_size] * 32767).astype(np.int16)
            yield pcm.tobytes()

    def unload(self) -> None:
        """
        Удаляет модель из памяти и очищает кэш CUDA.
        """
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("Silero TTS выгружена из VRAM")

    def warmup(self) -> None:
        """
        Выполняет тихий 'прогрев' модели тестовой фразой.
        """
        self.model.apply_tts(
            text="Прогрев",
            speaker=self.config.speaker_name,
            sample_rate=self.config.sample_rate,
        )
        logger.info("Silero TTS прогрет (тихо)")

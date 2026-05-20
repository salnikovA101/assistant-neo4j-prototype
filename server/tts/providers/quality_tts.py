import asyncio
import logging
import time
from typing import AsyncGenerator

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS

from server.utils.config import QualityTtsConfig, TtsConfig
from server.tts.base import BaseTTSProvider

logger = logging.getLogger(__name__)


class QualityTTSProvider(BaseTTSProvider):
    """
    Провайдер высококачественного синтеза речи на базе модели FasterQwen3TTS.

    Ориентирован на естественное звучание и клонирование голоса, используя
    возможности видеокарт NVIDIA для инференса в формате bfloat16.
    """

    def __init__(self, config: TtsConfig) -> None:
        """
        Инициализирует модель QwenTTS с загрузкой весов в видеопамять.

        Args:
            config (TtsConfig): Общий объект конфигурации, из которого извлекаются
                настройки секции 'quality'.
        """
        self.config: QualityTtsConfig = config.quality
        logger.info(f"Загрузка TTS {self.config.model}...")
        self.model = FasterQwen3TTS.from_pretrained(
            model_name=self.config.model,
            device=self.config.device,
            attn_implementation=self.config.attn_implementation,
            max_seq_len=self.config.max_seq_len,
            dtype=torch.bfloat16,
        )
        logger.info("QwenTTS готов")

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь и yield'ит PCM-чанки (int16, 24kHz).

        Генерация запускается в отдельном потоке, чанки передаются
        в asyncio через очередь.

        Args:
            text (str): Текст для синтеза.

        Yields:
            bytes: PCM-данные (int16, little-endian).
        """
        if not text:
            return

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _generate() -> None:
            start = time.perf_counter()
            first_chunk = True
            for audio_chunk, _, _ in self.model.generate_voice_clone_streaming(
                text=text,
                language=self.config.language,
                ref_audio=self.config.ref_voice,
                ref_text=self.config.ref_text,
                chunk_size=self.config.chunk_size,
                xvec_only=True,
            ):
                pcm = (np.asarray(audio_chunk, dtype=np.float32) * 32767).astype(
                    np.int16
                )
                loop.call_soon_threadsafe(queue.put_nowait, pcm.tobytes())
                if first_chunk:
                    logger.info(
                        f"TTS Time-to-First-Chunk: {time.perf_counter() - start:.3f}s (Qwen)"
                    )
                    first_chunk = False

            logger.info(f"TTS Total Time: {time.perf_counter() - start:.3f}s (Qwen)")
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(None, _generate)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def unload(self) -> None:
        """
        Удаляет модель из памяти и принудительно очищает кэш CUDA.
        """
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("FasterQwen3TTS выгружена из VRAM")

    def warmup(self) -> None:
        """
        Выполняет тихий 'прогрев' модели коротким тестовым словом.
        """
        for _ in self.model.generate_voice_clone_streaming(
            text="Прогрев",
            language=self.config.language,
            ref_audio=self.config.ref_voice,
            ref_text=self.config.ref_text,
            chunk_size=self.config.chunk_size,
            xvec_only=True,
        ):
            break
        logger.info("QwenTTS прогрет (тихо)")

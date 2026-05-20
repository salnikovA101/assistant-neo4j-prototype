import asyncio
import io
import logging
import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import soundfile as sf
from faster_whisper import WhisperModel

from server.utils.config import SttConfig
from server.utils.tracing import (
    OI_INPUT_VALUE,
    OI_SPAN_KIND,
    OISpanKind,
    get_tracer,
    set_span_error,
    set_span_ok,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class STTProvider:
    """
    Класс для обеспечения работы системы распознавания речи (Speech-to-Text)
    на базе модели Faster Whisper.
    """

    def __init__(self, config: SttConfig) -> None:
        """
        Инициализирует модель Whisper с заданными параметрами конфигурации.

        Args:
            config (SttConfig): Объект конфигурации, содержащий параметры модели.
        """
        self.config: SttConfig = config
        logger.info(f"Загрузка Whisper {self.config.model} на {self.config.device}...")
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self._warmup()
        logger.info("Whisper готов")

    def _warmup(self) -> None:
        """
        Выполняет 'прогрев' модели путем обработки пустого аудиосигнала.
        Это необходимо для инициализации весов и кэша, чтобы последующие
        реальные запросы обрабатывались без задержек.
        """
        logger.debug("Прогрев STT модели...")
        start = time.perf_counter()
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.model.transcribe(dummy_audio, beam_size=5, language=self.config.language)
        logger.debug(f"Прогрев STT завершен за {time.perf_counter() - start:.3f} сек")

    async def transcribe(self, audio: npt.NDArray[np.float32]) -> Optional[str]:
        """
        Асинхронно преобразует аудиопоток в текст.

        Args:
            audio (npt.NDArray[np.float32]): Аудиоданные в формате массива NumPy (float32).

        Returns:
            str: Распознанный текст. Возвращает None, если входной массив пуст.
        """
        if len(audio) == 0:
            return None

        with tracer.start_as_current_span("stt_transcribe") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, f"Audio array, shape: {audio.shape}")

            start = time.perf_counter()
            try:
                text = await asyncio.to_thread(self._transcribe_sync, audio)
                elapsed = time.perf_counter() - start

                text = text.strip()
                logger.info(f"STT Time: {elapsed:.3f}s | Result: {text}")

                set_span_ok(span, text)
                return text
            except Exception as e:
                set_span_error(span, str(e))
                raise

    async def transcribe_bytes(self, wav_bytes: bytes) -> Optional[str]:
        """
        Принимает WAV bytes с клиента, декодирует в float32 numpy и транскрибирует.

        Args:
            wav_bytes (bytes): Сырые байты WAV-файла (16kHz, mono, float32).

        Returns:
            str: Распознанный текст, или None если аудио пустое.
        """
        audio_np, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        return await self.transcribe(audio_np.astype(np.float32))

    def _transcribe_sync(self, audio: npt.NDArray[np.float32]) -> str:
        """
        Внутренний синхронный метод для выполнения транскрибации.

        Использует VAD (Voice Activity Detection) для фильтрации тишины и
        ограничивает распознавание заданным языком.

        Args:
            audio (npt.NDArray[np.float32]): Аудиоданные для обработки.

        Returns:
            str: Склеенный текст всех распознанных сегментов аудио.
        """
        segments, _ = self.model.transcribe(
            audio, beam_size=5, vad_filter=True, language=self.config.language
        )
        return " ".join(seg.text for seg in segments)

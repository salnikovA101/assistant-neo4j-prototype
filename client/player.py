import logging
from typing import AsyncIterator, Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class StreamingPlayer:
    """
    Воспроизводит PCM int16 24kHz чанки в реальном времени через sounddevice.

    Используется клиентом для проигрывания аудио, полученного от сервера
    в виде стрима байтов.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None

    def _ensure_stream(self) -> None:
        """Открывает аудиопоток при первом чанке."""
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=2048,
                latency="high",
            )
            self._stream.start()

    def feed(self, pcm_bytes: bytes) -> None:
        """
        Принимает PCM-чанк (int16) и воспроизводит его.

        Args:
            pcm_bytes (bytes): PCM-данные (int16, little-endian, mono).
        """
        if not pcm_bytes:
            return
        self._ensure_stream()
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32767.0
        self._stream.write(audio.reshape(-1, 1))  # type: ignore

    def drain(self) -> None:
        """Закрывает аудиопоток после окончания воспроизведения."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Ошибка при закрытии аудиопотока: {e}")
            finally:
                self._stream = None

    async def play_stream(
        self,
        chunk_iter: AsyncIterator[bytes],
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """
        Воспроизводит стрим PCM-чанков.

        Args:
            chunk_iter: Асинхронный итератор байтовых чанков.
            should_stop: Функция, возвращающая True для мгновенной остановки (barge-in).
        """
        try:
            async for chunk in chunk_iter:
                if should_stop and should_stop():
                    logger.info("Воспроизведение прервано пользователем (barge-in)")
                    break
                self.feed(chunk)
        finally:
            self.drain()

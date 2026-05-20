import asyncio
import io
import logging
from typing import List

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf

from client.audio_cues import AudioCues

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Класс для управления захватом аудио через микрофон с использованием механики Push-to-Talk.

    Обеспечивает ожидание нажатия горячей клавиши, циклическую запись аудиоданных
    в буфер и последующую склейку в единый массив для обработки STT.
    """

    def __init__(self, ptt_key: str, samplerate: int = 16000) -> None:
        """
        Инициализирует рекордер и настраивает параметры захвата.

        Args:
            ptt_key (str): Клавиша активации Push-to-Talk.
            samplerate (int, optional): Частота дискретизации аудио. По умолчанию 16000 Гц.
        """
        self.ptt_key = ptt_key
        self.samplerate: int = samplerate
        self.recording: bool = False
        self.chunks: List[np.ndarray] = []
        self.cues = AudioCues()

    def _callback(self, indata: np.ndarray, frames, time_info, status):
        """
        Внутренний обработчик (callback) для входящего аудиопотока.
        """
        if self.recording:
            self.chunks.append(indata.copy())

    async def record(self) -> bytes:
        """
        Запускает и останавливает запись аудио по удержанию клавиши.

        Returns:
            bytes: Сконвертированные WAV-аудиоданные.
        """
        self.chunks.clear()

        self.cues.play_start()
        logger.info(f"Отпусти '{self.ptt_key}', чтобы закончить")
        self.recording = True

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self._callback,
            dtype="float32",
        ):
            while keyboard.is_pressed(self.ptt_key):
                await asyncio.sleep(0.03)

        self.recording = False
        self.cues.play_stop()

        if not self.chunks:
            return b""

        audio = np.concatenate(self.chunks, axis=0).flatten().astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, audio, self.samplerate, format="WAV", subtype="FLOAT")
        return buf.getvalue()

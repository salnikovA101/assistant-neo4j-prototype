import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCues:
    """
    Генератор коротких звуковых сигналов для обратной связи Push-to-Talk.

    Создаёт два тона при инициализации (восходящий и нисходящий sweep)
    и хранит их как numpy-массивы для мгновенного воспроизведения.
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        """
        Инициализирует и предгенерирует звуковые сигналы.

        Args:
            sample_rate (int): Частота дискретизации аудио. По умолчанию 44100 Гц.
        """
        self.sample_rate = sample_rate
        self._start_tone = self._generate_sweep(
            freq_start=600, freq_end=900, duration=0.12
        )
        self._stop_tone = self._generate_sweep(
            freq_start=900, freq_end=600, duration=0.12
        )
        logger.debug("Звуковые сигналы PTT сгенерированы")

    def _generate_sweep(
        self, freq_start: float, freq_end: float, duration: float
    ) -> np.ndarray:
        """
        Генерирует линейный частотный sweep с fade-in/fade-out.
        """
        n_samples = int(self.sample_rate * duration)
        np.linspace(0, duration, n_samples, endpoint=False)

        freqs = np.linspace(freq_start, freq_end, n_samples)
        phase = 2 * np.pi * np.cumsum(freqs) / self.sample_rate
        signal = 0.3 * np.sin(phase)

        fade_len = n_samples // 10
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        signal[:fade_len] *= fade_in
        signal[-fade_len:] *= fade_out

        return signal.astype(np.float32)

    def play_start(self) -> None:
        """Воспроизводит восходящий тон (начало записи)."""
        try:
            sd.play(self._start_tone, self.sample_rate)
        except Exception as e:
            logger.warning(f"Не удалось воспроизвести сигнал начала записи: {e}")

    def play_stop(self) -> None:
        """Воспроизводит нисходящий тон (конец записи)."""
        try:
            sd.play(self._stop_tone, self.sample_rate)
        except Exception as e:
            logger.warning(f"Не удалось воспроизвести сигнал конца записи: {e}")

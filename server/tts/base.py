from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseTTSProvider(ABC):
    """
    Абстрактный базовый класс для всех провайдеров синтеза речи (TTS).

    Определяет обязательный интерфейс, который должен реализовать каждый
    конкретный провайдер.
    """

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь из текста и возвращает PCM-чанки (int16, 24kHz, mono).

        Используется в серверном режиме — без воспроизведения через sounddevice.

        Args:
            text (str): Текст для синтеза речи.

        Yields:
            bytes: PCM-чанки (int16, little-endian).
        """
        yield b""  # pragma: no cover

    @abstractmethod
    def unload(self) -> None:
        """
        Выгружает модель из памяти и освобождает ресурсы.

        Метод вызывается при переключении между провайдерами или при
        завершении работы приложения для очистки VRAM/RAM.
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """
        Выполняет предварительную подготовку (прогрев) модели.

        Включает в себя прогон через модель короткого текста,
        чтобы инициализировать веса и исключить задержку при первом реальном запросе.
        """
        pass

from enum import StrEnum


class LLMProviderType(StrEnum):
    """
    Тип провайдера для LLM.
    """

    OPENAI = "openai"


class TTSModes(StrEnum):
    """
    Режимы работы системы синтеза речи (TTS).
    """

    SPEED = "speed"
    QUALITY = "quality"
    CLOUD = "cloud"

from enum import StrEnum


class LLMProviderType(StrEnum):
    """
    Тип провайдера для LLM.
    """

    OPENAI = "openai"


class EmbeddingBackend(StrEnum):
    """
    Backend for graph retrieval embeddings (must match Neo4j vector dims).
    """

    TEI = "tei"  # local text-embeddings-inference
    OPENROUTER = "openrouter"  # OpenRouter / remote OpenAI-compatible API


class TTSModes(StrEnum):
    """
    Режимы работы системы синтеза речи (TTS).
    """

    SPEED = "speed"
    QUALITY = "quality"
    CLOUD = "cloud"

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

    OLLAMA = "ollama"  # Local Ollama OpenAI-compatible /v1/embeddings


class TTSModes(StrEnum):
    """
    Режимы работы системы синтеза речи (TTS).
    """

    SPEED = "speed"
    QUALITY = "quality"
    CLOUD = "cloud"

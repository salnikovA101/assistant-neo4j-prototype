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


# Written on retrieval snapshots. Older checkpoints used "v6-checkpoint-1".
RETRIEVAL_STATE_VERSION = "retrieval-carousel-v1"
LEGACY_RETRIEVAL_STATE_VERSIONS = frozenset({"v6-checkpoint-1"})

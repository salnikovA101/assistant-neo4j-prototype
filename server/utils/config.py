from pathlib import Path
import logging

import yaml

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from server.utils.constants import EmbeddingBackend, LLMProviderType, TTSModes


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"


class GraphEmbeddingsConfig(BaseModel):
    """Embeddings for Neo4j vector search (must match index dimensions)."""

    backend: EmbeddingBackend = EmbeddingBackend.OPENROUTER
    model: str = "nvidia/nemotron-3-embed-1b:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    max_client_batch_size: int = 32
    max_input_chars: int = 8192


class OpenAIProfile(BaseModel):
    provider: LLMProviderType = LLMProviderType.OPENAI
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_output_tokens: int = 4096
    context_length: int = 4096
    max_turns: int = 2
    think: bool = False
    # Gemma/LM Studio: inject into system message so thinking stays on after tool results.
    # Empty for DeepSeek/OpenRouter (they use reasoning API params instead).
    think_token: str = ""
    # OpenRouter/DeepSeek/OpenAI effort when think=true: low|medium|high|max
    think_effort: str = "high"


class LlmProfiles(BaseModel):
    gemini: OpenAIProfile = Field(default_factory=OpenAIProfile)
    other: OpenAIProfile = Field(default_factory=OpenAIProfile)
    lm_studio: OpenAIProfile = Field(default_factory=OpenAIProfile)
    ollama: OpenAIProfile = Field(default_factory=OpenAIProfile)


class LlmConfig(BaseModel):
    current_profile: str = "other"
    cypher_profile: str = "other"
    # Nested LLM for mock_decompose / tools that need a second profile
    tool_profile: str = "other"
    history_len: int = 6
    prompt_folder: str = "prompts"
    profiles: LlmProfiles = Field(default_factory=LlmProfiles)


class SttConfig(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_bfloat16"
    language: str = "ru"


class SpeedTtsConfig(BaseModel):
    silero_speaker: str = "baya"
    sample_rate: int = 24000
    language: str = "ru"
    speaker_type: str = "v5_ru"
    device: str = "cuda"
    speaker_name: str = "baya"


class QualityTtsConfig(BaseModel):
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 2048
    ref_voice: str = "voices/example.wav"
    ref_text: str = ""
    language: str = "Russian"
    chunk_size: int = 4


class CloudTtsConfig(BaseModel):
    model: str = "openai/gpt-4o-mini-tts-2025-12-15"
    voice: str = "alloy"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"


class TtsConfig(BaseModel):
    mode: TTSModes = TTSModes.QUALITY
    speed: SpeedTtsConfig = Field(default_factory=SpeedTtsConfig)
    quality: QualityTtsConfig = Field(default_factory=QualityTtsConfig)
    cloud: CloudTtsConfig = Field(default_factory=CloudTtsConfig)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    llm_timeout: int = 300


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    debug_mode: bool = False
    audio_enabled: bool = True
    server: ServerConfig = Field(default_factory=ServerConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    graph_embeddings: GraphEmbeddingsConfig = Field(default_factory=GraphEmbeddingsConfig)
    run_id: str = ""
    limit: int = 50


def load_config() -> AppConfig:
    """Загружает конфиг из server/config.yaml + .env переменных."""

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.getLogger(__name__).warning(
            f"{config_path} не найден, используются значения по умолчанию"
        )
        data = {}

    return AppConfig(**data)

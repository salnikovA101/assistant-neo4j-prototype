from pathlib import Path
import logging

import yaml

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from server.utils.constants import LLMProviderType, TTSModes

_REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_PROFILE = "auto"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"


class OpenAIProfile(BaseModel):
    provider: LLMProviderType = LLMProviderType.OPENAI
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_output_tokens: int = 4096
    context_window: int = 32768
    max_turns: int = 2
    think: bool = False
    # Gemma/LM Studio: inject into system message so thinking stays on after tool results.
    # Empty for DeepSeek/OpenRouter (they use reasoning API params instead).
    think_token: str = ""
    # Default UI/API effort when think=true (must be in think_efforts when set).
    think_effort: str = "high"
    # Per-model menu: UI renders these strings as-is. Empty → [think_effort].
    think_efforts: list[str] = Field(default_factory=list)
    # Qwen3.8 / Qwen Cloud: keep <think> from prior turns (tool loop).
    # Official Qwen3.8 default is True; Qwen Cloud reads extra_body.preserve_thinking.
    preserve_thinking: bool = False
    # Label in the UI model picker. Empty → profile id.
    display_name: str = ""
    # How to encode thinking for this gateway: qwen38, qwen37, deepseek_v4, glm, kimi.
    # Empty → generic OpenAI / Ollama extras.
    think_family: str = ""


class LlmProfiles(BaseModel):
    gemini: OpenAIProfile = Field(default_factory=OpenAIProfile)
    other: OpenAIProfile = Field(default_factory=OpenAIProfile)
    lm_studio: OpenAIProfile = Field(default_factory=OpenAIProfile)
    ollama: OpenAIProfile = Field(default_factory=OpenAIProfile)
    ollama_gptoss: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen_cloud: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen38_flash: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen38_max: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen38_27b: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen38_2_4t: OpenAIProfile = Field(default_factory=OpenAIProfile)
    deepseek_v4_pro: OpenAIProfile = Field(default_factory=OpenAIProfile)
    kimi_k3: OpenAIProfile = Field(default_factory=OpenAIProfile)
    glm_52: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen37_max: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen37_plus: OpenAIProfile = Field(default_factory=OpenAIProfile)
    qwen37_flash: OpenAIProfile = Field(default_factory=OpenAIProfile)


class LlmConfig(BaseModel):
    current_profile: str = "other"
    # Nested LLM for mock_decompose / tools that need a second profile
    tool_profile: str = "other"
    # Profiles shown in the UI picker. Empty → [current_profile].
    ui_profiles: list[str] = Field(default_factory=list)
    # Auto rotates through these DashScope ids as each model's free quota dies.
    auto_order: list[str] = Field(default_factory=list)
    fallback_profile: str = "ollama"
    # In-memory HistoryManager maxlen for audio/fallback only. Live chat uses
    # the full checkpoint lineage and is not truncated by this value.
    history_len: int = 10000
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
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    debug_mode: bool = False
    audio_enabled: bool = True
    # Neo4j relationship.run_id for ANN/bridges. Empty = no corpus filter.
    run_id: str = ""
    # S2b cross-encoder. false skips Ettin (ANN sim order).
    rerank_enabled: bool = True
    staged_enabled: bool = True
    cards_enabled: bool = True
    # Account database and browser-session policy.
    app_db_path: str = "data/assistant.db"
    auth_cookie_secure: bool = False
    auth_session_days: int = 30
    auth_trusted_origins: str = ""
    server: ServerConfig = Field(default_factory=ServerConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)


def retrieval_param_overrides(config: AppConfig | None = None) -> dict:
    """Params fields driven by config.yaml (run_id, rerank_enabled)."""
    cfg = config or load_config()
    return {
        "run_id": (cfg.run_id or "").strip(),
        "rerank_enabled": bool(cfg.rerank_enabled),
    }


def llm_profile(config: LlmConfig, name: str) -> OpenAIProfile | None:
    """Return a named LLM profile, or None if the id is unknown."""
    key = (name or "").strip()
    if not key or key == AUTO_PROFILE:
        return None
    return getattr(config.profiles, key, None)


def ui_selectable_profiles(llm: LlmConfig) -> list[str]:
    """Profile ids the web UI may send. Unknown yaml names are dropped."""
    raw = [str(item).strip() for item in (llm.ui_profiles or []) if str(item).strip()]
    names = [
        name
        for name in raw
        if name == AUTO_PROFILE or llm_profile(llm, name) is not None
    ]
    if names:
        return names
    current = (llm.current_profile or "").strip()
    if current == AUTO_PROFILE:
        return [AUTO_PROFILE]
    if current and llm_profile(llm, current) is not None:
        return [current]
    return []


def boot_profile_name(llm: LlmConfig) -> str:
    """Concrete yaml profile used to construct a provider at process start."""
    current = (llm.current_profile or "").strip()
    if current == AUTO_PROFILE:
        for name in llm.auto_order or []:
            profile = llm_profile(llm, name)
            if profile is not None and (profile.model or "").strip():
                return name
        fallback = (llm.fallback_profile or "").strip()
        if fallback and llm_profile(llm, fallback) is not None:
            return fallback
    if current and llm_profile(llm, current) is not None:
        return current
    selectable = [name for name in ui_selectable_profiles(llm) if name != AUTO_PROFILE]
    if selectable:
        return selectable[0]
    raise ValueError("No LLM profile configured")


_YAML_TOP_LEVEL_KEYS = {
    "debug_mode",
    "audio_enabled",
    "run_id",
    "rerank_enabled",
    "staged_enabled",
    "cards_enabled",
    "app_db_path",
    "auth_cookie_secure",
    "auth_session_days",
    "auth_trusted_origins",
    "server",
    "stt",
    "tts",
    "llm",
    "neo4j",
}
_PLACEHOLDER_NEO4J_PASSWORDS = frozenset({"", "password123"})


def resolve_request_profile(llm: LlmConfig, name: str | None) -> str:
    """Accept a UI profile id, else current_profile. Unknown requested id is an error."""
    requested = (name or "").strip()
    allowed = set(ui_selectable_profiles(llm))
    if requested:
        if requested not in allowed:
            raise ValueError("unknown_profile")
        return requested
    current = (llm.current_profile or "").strip()
    if current == AUTO_PROFILE:
        return AUTO_PROFILE
    if current and llm_profile(llm, current) is not None:
        return current
    selectable = ui_selectable_profiles(llm)
    if selectable:
        return selectable[0]
    raise ValueError("No LLM profile configured")


def validate_runtime_config(config: AppConfig) -> None:
    """Refuse to boot with placeholder Neo4j credentials or an empty LLM model."""
    password = (config.neo4j.password or "").strip()
    if password in _PLACEHOLDER_NEO4J_PASSWORDS:
        raise ValueError(
            "NEO4J__PASSWORD is missing or still the placeholder. "
            "Set it in the repo-root .env."
        )
    profile_name = boot_profile_name(config.llm)
    profile = llm_profile(config.llm, profile_name)
    if profile is None or not (profile.model or "").strip():
        raise ValueError("current_profile must name an LLM profile with a model")


def inherit_ollama_cloud_credentials(config: AppConfig) -> None:
    """Copy Ollama Cloud api_key/base_url onto sibling ollama_* profiles."""
    source = getattr(config.llm.profiles, "ollama", None)
    if source is None:
        return
    source_key = (source.api_key or "").strip()
    source_url = (source.base_url or "").strip()
    if not source_key and not source_url:
        return
    for name in type(config.llm.profiles).model_fields:
        if name == "ollama" or not name.startswith("ollama"):
            continue
        profile = getattr(config.llm.profiles, name, None)
        if profile is None:
            continue
        if source_key and not (profile.api_key or "").strip():
            profile.api_key = source_key
        if source_url and not (profile.base_url or "").strip():
            profile.base_url = source_url


def inherit_qwen_cloud_credentials(config: AppConfig) -> None:
    """Copy DashScope api_key/base_url onto Auto-catalog profiles."""
    source = getattr(config.llm.profiles, "qwen_cloud", None)
    if source is None:
        return
    source_key = (source.api_key or "").strip()
    source_url = (source.base_url or "").strip()
    if not source_key and not source_url:
        return
    names = list(config.llm.auto_order or [])
    for name in names:
        profile = llm_profile(config.llm, name)
        if profile is None:
            continue
        if source_key and not (profile.api_key or "").strip():
            profile.api_key = source_key
        if source_url and not (profile.base_url or "").strip():
            profile.base_url = source_url


def load_config() -> AppConfig:
    """Загружает конфиг из server/config.yaml + .env переменных."""

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"{config_path} не найден") from None
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must be a mapping")
    unknown = sorted(set(data) - _YAML_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"{config_path} unknown keys: {', '.join(unknown)}")

    config = AppConfig(**data)
    inherit_ollama_cloud_credentials(config)
    inherit_qwen_cloud_credentials(config)
    return config

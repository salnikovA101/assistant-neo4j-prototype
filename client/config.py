from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ClientConfig(BaseSettings):
    """
    Конфигурация клиента голосового ассистента.

    Загружается из client/config.yaml, значения можно переопределить
    через переменные окружения с префиксом CLIENT__.
    """

    model_config = SettingsConfigDict(env_prefix="CLIENT__")

    server_url: str = "http://localhost:8000"
    push_to_talk_key: str = "right ctrl"
    sample_rate: int = 16000
    playback_sample_rate: int = 24000
    debug_mode: bool = False
    enable_voice_input: bool = True
    enable_text_input: bool = True
    request_timeout: int = 300


def load_config() -> ClientConfig:
    """Загружает конфиг клиента из client/config.yaml."""
    import logging

    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.getLogger(__name__).warning(
            f"{config_path} не найден, используются значения по умолчанию"
        )
        data = {}

    return ClientConfig(**data)

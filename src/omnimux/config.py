from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    port: int = 11444

    sqlite_db_path: str = "omnimux_state.db"
    models_json_path: str = "models.json"
    default_cooldown_minutes: int = 60
    image_token_reserve: int = 4096
    proxy_password: str | None = None
    default_key_concurrency: int = 10
    retry_jitter_min_ms: int = 50
    retry_jitter_max_ms: int = 250
    sse_keepalive_interval_seconds: float = 3.0
    session_affinity_ttl_seconds: float = 300.0
    max_session_affinity_entries: int = 10000


settings = Settings()

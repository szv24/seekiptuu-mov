from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_path: Path = Path(__file__).resolve().parent.parent / "movies.db"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    ollama_timeout: float = 120.0
    log_level: str = "INFO"

    model_config = {"env_prefix": "MOVIE_AGENT_"}


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str
    model_name: str = "gemini-3.6-flash"
    max_iterations: int = 10
    max_tool_result_chars: int = 8000  # обрізання результатів tools


settings = Settings()
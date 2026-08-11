from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str
    model_name: str = "gemini-3.6-flash"
    max_iterations: int = 10
    max_tool_result_chars: int = 8000
    output_dir: str = "output"
    max_history_chars: int = 40000

    data_dir: str = "data"
    index_dir: str = "index"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    knowledge_search_top_k: int = 5
    rrf_k: int = 60
    candidate_pool_multiplier: int = 4
    web_search_snippet_limit: int = 300


settings = Settings()
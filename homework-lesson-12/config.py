from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict
from langfuse import get_client


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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

    # Multi-agent (lesson-8)
    max_revision_rounds: int = 2


settings = Settings()

# --- Системні промпти агентів завантажуються з Langfuse Prompt Management ---
# Жодного хардкоду тексту промптів у коді — лише назва та label.

_langfuse = get_client()

PLANNER_SYSTEM_PROMPT = _langfuse.get_prompt("planner-system-prompt", label="production").compile()
RESEARCHER_SYSTEM_PROMPT = _langfuse.get_prompt("researcher-system-prompt", label="production").compile()
CRITIC_SYSTEM_PROMPT = _langfuse.get_prompt("critic-system-prompt", label="production").compile()
SUPERVISOR_SYSTEM_PROMPT = _langfuse.get_prompt(
    "supervisor-system-prompt", label="production"
).compile(max_revision_rounds=settings.max_revision_rounds)
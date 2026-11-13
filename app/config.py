from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    # Groq production model, checked 2026-11-02 (console.groq.com/docs/models).
    # llama-3.1-8b-instant was shut down for free/developer tiers on 2026-08-16;
    # Groq lists openai/gpt-oss-20b as its replacement.
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    database_url: str = "sqlite+aiosqlite:///./meridian.db"
    embedder_model: str = "all-MiniLM-L6-v2"
    # "cpu" by default: on Apple Silicon the default MPS backend crashed the
    # server (segfault) on the first request. Set to "" to let
    # sentence-transformers pick a device, or e.g. "cuda".
    embedder_device: str = "cpu"
    top_k_passages: int = 10
    # "sentence-transformers" (needs requirements-ml.txt) or "hashing" (lexical, no model).
    embedder_backend: str = "sentence-transformers"
    # Public demo: serve the one-page demo and the stateless /api/research stream
    # instead of the session API, and simulate the LLM stages when no Groq key is set.
    demo_mode: bool = False
    demo_rate_limit_per_hour: int = 12
    demo_max_query_chars: int = 200

    @property
    def simulate_llm(self) -> bool:
        return self.demo_mode and not self.groq_api_key

    class Config:
        env_file = ".env"


settings = Settings()

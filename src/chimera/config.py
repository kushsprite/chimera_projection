"""Central configuration.

All paths resolve relative to the project root, never the current working
directory, so code behaves the same whether it runs from the repo root,
from notebooks/, or from scripts/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None


def _find_project_root() -> Path:
    override = os.environ.get("CHIMERA_ROOT")
    if override:
        return Path(override).resolve()
    # src/chimera/config.py -> project root is two levels above src/
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _find_project_root()

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    cache: Path = PROJECT_ROOT / "data" / "cache"
    fixtures: Path = PROJECT_ROOT / "data" / "fixtures"
    raw_matches: Path = PROJECT_ROOT / "data" / "raw" / "IPL Match Data"

    @property
    def history_csv(self) -> Path:
        return self.processed / "player_match_features.csv"

    @property
    def new_matches_csv(self) -> Path:
        # rows appended by scripts/update_history.py; never touches the training CSV
        return self.processed / "new_matches.csv"

    @property
    def credits_csv(self) -> Path:
        return self.processed / "player_credits.csv"

    @property
    def city_coords_json(self) -> Path:
        return self.processed / "city_coords.json"

    @property
    def weather_lookup_json(self) -> Path:
        return self.processed / "weather_lookup_final.json"

    @property
    def weather_live_cache_json(self) -> Path:
        return self.cache / "weather_live_cache.json"

    @property
    def lgbm_model(self) -> Path:
        return self.models / "lgbm_final.pkl"

    @property
    def lstm_model(self) -> Path:
        return self.models / "lstm_final.pt"

    @property
    def ensemble_config(self) -> Path:
        return self.models / "ensemble_config.json"


@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)

    # inference
    device: str = os.environ.get("CHIMERA_DEVICE", "cpu")  # "cpu", "mps", "cuda", or "auto"

    # LLM generation, tried in order until one works
    groq_models: tuple = tuple(
        m.strip() for m in os.environ.get(
            "CHIMERA_GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"
        ).split(",") if m.strip()
    )
    gemini_generation_model: str = os.environ.get("CHIMERA_GEMINI_GEN_MODEL", "gemini-flash-latest")

    # embeddings
    gemini_embedding_model: str = os.environ.get("CHIMERA_GEMINI_EMBED_MODEL", "gemini-embedding-2")

    # RAG retrieval
    rag_chunk_words: int = 300
    rag_chunk_overlap: int = 50
    rag_min_chunk_words: int = 50
    rag_top_k: int = 6
    rag_lookback_days: int = 7
    rag_doc_cache_ttl_hours: float = 3.0
    rag_max_fulltext_fetches: int = 6

    @property
    def gemini_api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    @property
    def groq_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY")

    @property
    def gnews_api_key(self) -> str | None:
        return os.environ.get("GNEWS_API_KEY")


def get_settings() -> Settings:
    s = Settings()
    s.paths.cache.mkdir(parents=True, exist_ok=True)
    return s

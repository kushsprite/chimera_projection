import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Tests must never hit real paid/rate-limited APIs.
for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "GNEWS_API_KEY"):
    os.environ.pop(key, None)

DATA_READY = (ROOT / "data/processed/player_match_features.csv").exists() and \
             (ROOT / "models/lgbm_final.pkl").exists() and (ROOT / "models/lstm_final.pt").exists()
needs_data = pytest.mark.skipif(not DATA_READY, reason="needs data/processed and models/ (gitignored)")


@pytest.fixture(scope="session")
def engine():
    if not DATA_READY:
        pytest.skip("needs data and models")
    from chimera.config import get_settings
    from chimera.engine import ChimeraEngine
    from chimera.rag.explainer import Explainer
    from chimera.rag.llm import LLMRouter

    eng = ChimeraEngine(get_settings(), explainer=Explainer(get_settings(), llm=LLMRouter([])))
    eng.weather.coords = lambda city: None  # no geocoding / forecast calls in tests
    return eng

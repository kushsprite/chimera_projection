"""Check every live connection with your real keys. Run this first on a new machine.

    python scripts/smoke_test_live.py
    python scripts/smoke_test_live.py --url https://www.cricbuzz.com/some-preview   # also test article extraction

Each check prints PASS, FAIL (with the reason), or SKIP (no key set).
Nothing here is required: every failure has a fallback in the real pipeline.
"""
import argparse
from datetime import date, timedelta

import _bootstrap  # noqa: F401

from chimera.config import get_settings

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        status = "SKIP" if isinstance(detail, str) and detail.startswith("SKIP") else "PASS"
        RESULTS.append((status, name, detail.replace("SKIP: ", "") if status == "SKIP" else detail))
    except Exception as e:
        RESULTS.append(("FAIL", name, f"{type(e).__name__}: {str(e)[:160]}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="an article URL to test full-text extraction")
    args = ap.parse_args()
    s = get_settings()

    def weather():
        from chimera.weather import FORECAST_URL, WeatherService
        ws = WeatherService(s, {})
        latlon = ws.coords("Mumbai")
        vals = ws._fetch(FORECAST_URL, *latlon, (date.today() + timedelta(days=1)).isoformat())
        assert vals, "no hourly data returned"
        return f"Mumbai tomorrow evening: {vals['temp']:.1f}C, {vals['humidity']:.0f}% humidity"

    def google_news():
        from chimera.rag.sources import GoogleNewsSource, MatchQuery
        q = MatchQuery("Mumbai Indians", "Chennai Super Kings", "Wankhede", "Mumbai", date.today(), 30)
        docs = GoogleNewsSource(fetch_full_text=False).fetch(q)
        assert docs, "no results (Google may be rate limiting; try again later)"
        return f"{len(docs)} headlines, e.g. '{docs[0].title[:70]}'"

    def gnews():
        if not s.gnews_api_key:
            return "SKIP: GNEWS_API_KEY not set (optional, free at gnews.io)"
        from chimera.rag.sources import GNewsSource, MatchQuery
        q = MatchQuery("Mumbai Indians", "Chennai Super Kings", "Wankhede", "Mumbai", date.today(), 30)
        docs = GNewsSource(s.gnews_api_key, max_queries=1, fetch_full_text=False).fetch(q)
        return f"{len(docs)} articles" + (f", e.g. '{docs[0].title[:60]}'" if docs else " (none in the last 30 days)")

    def url_extract():
        if not args.url:
            return "SKIP: pass --url to test"
        from chimera.rag.sources import fetch_url_text
        d = fetch_url_text(args.url)
        assert d, "could not extract article text (site may block scripts)"
        return f"{len(d.text.split())} words from '{(d.title or '')[:60]}'"

    def gemini_embed():
        if not s.gemini_api_key:
            return "SKIP: GEMINI_API_KEY not set (retrieval will use TF-IDF)"
        from chimera.rag.embeddings import EmbeddingCache, GeminiEmbedder
        e = GeminiEmbedder(s.gemini_api_key, s.gemini_embedding_model,
                           EmbeddingCache(s.paths.cache / "embeddings.sqlite"))
        v = e.embed(["The pitch at Eden Gardens is expected to assist spinners."])
        return f"{e._resolved_model or e.model}: {v.shape[1]}-dim vector"

    def groq():
        if not s.groq_api_key:
            return "SKIP: GROQ_API_KEY not set"
        from chimera.rag.llm import GroqGenerator
        r = GroqGenerator(s.groq_api_key, s.groq_models).generate("Reply with exactly: OK", max_tokens=200)
        return f"{r.model} replied '{r.text[:30]}'"

    def gemini_gen():
        if not s.gemini_api_key:
            return "SKIP: GEMINI_API_KEY not set"
        from chimera.rag.llm import GeminiGenerator
        r = GeminiGenerator(s.gemini_api_key, s.gemini_generation_model).generate("Reply with exactly: OK", 200)
        return f"{r.model} replied '{r.text[:30]}'"

    for name, fn in [("Open-Meteo weather forecast", weather), ("Google News RSS", google_news),
                     ("GNews API", gnews), ("Article text extraction", url_extract),
                     ("Gemini embeddings", gemini_embed), ("Groq generation", groq),
                     ("Gemini generation (fallback)", gemini_gen)]:
        check(name, fn)

    width = max(len(n) for _, n, _ in RESULTS)
    for status, name, detail in RESULTS:
        print(f"[{status:4s}] {name:<{width}}  {detail}")
    if any(r[0] == "FAIL" for r in RESULTS):
        print("\nFailures fall back gracefully, but explanations will be weaker without them.")


if __name__ == "__main__":
    main()

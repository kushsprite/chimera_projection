"""RAG components, tested against responses shaped like the real services."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chimera.rag import sources as S
from chimera.rag.chunking import chunk_documents, chunk_text
from chimera.rag.embeddings import EmbeddingCache, GeminiEmbedder, TfidfEmbedder
from chimera.rag.explainer import DISCLAIMER, Explainer, RagOptions
from chimera.rag.llm import GroqGenerator, LLMError, LLMRouter, RateLimited
from chimera.rag.retriever import VectorIndex

RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>
<item><title>RCB vs KKR preview: dew expected to play a big role at Chinnaswamy - ESPNcricinfo</title>
<link>https://news.google.com/rss/articles/CBMiAAA?oc=5</link>
<pubDate>Wed, 08 Apr 2026 10:00:00 GMT</pubDate>
<description>&lt;a href="x"&gt;RCB vs KKR preview: dew expected to play a big role at Chinnaswamy&lt;/a&gt;&amp;nbsp;&lt;font&gt;ESPNcricinfo&lt;/font&gt;</description>
<source url="https://www.espncricinfo.com">ESPNcricinfo</source></item>
<item><title>Hazlewood doubtful for KKR clash with shoulder niggle - Cricbuzz</title>
<link>https://news.google.com/rss/articles/CBMiBBB?oc=5</link>
<pubDate>Thu, 09 Apr 2026 08:00:00 GMT</pubDate>
<description>Hazlewood doubtful</description><source url="https://www.cricbuzz.com">Cricbuzz</source></item>
<item><title>Stock market rallies on Tuesday - Mint</title><link>https://news.google.com/rss/articles/CBMiCCC</link>
<pubDate>Tue, 07 Apr 2026 08:00:00 GMT</pubDate><description>markets</description><source url="x">Mint</source></item>
<item><title>RCB vs KKR head to head from 2019 - Someone</title><link>https://news.google.com/rss/articles/OLD</link>
<pubDate>Mon, 01 Apr 2019 08:00:00 GMT</pubDate><description>old</description><source url="x">Someone</source></item>
</channel></rss>"""

GNEWS = {"totalArticles": 2, "articles": [
    {"title": "Chinnaswamy pitch report for RCB vs KKR", "description": "Flat deck, short boundaries.",
     "content": "The Chinnaswamy surface is expected to be flat with short boundaries favouring hitters... [2311 chars]",
     "url": "https://example.com/pitch", "publishedAt": "2026-04-09T06:00:00Z",
     "source": {"name": "Example Sports", "url": "https://example.com"}},
    {"title": "No title body", "description": None, "content": None, "url": "https://example.com/2",
     "publishedAt": "2026-04-09T07:00:00Z", "source": {"name": "Example"}},
]}

ARTICLE_HTML = """<html><head><title>Preview | Site</title>
<script type="application/ld+json">{"@type":"NewsArticle","articleBody":"%s"}</script></head>
<body><nav><p>Menu item that is long enough to count as a paragraph normally</p></nav>
<p>This paragraph talks about the Chinnaswamy pitch and how dew will help the chasing side tonight.</p>
</body></html>""" % ("The pitch at the Chinnaswamy is flat and dew is expected later in the evening. " * 5)

QUERY = S.MatchQuery("Royal Challengers Bengaluru", "Kolkata Knight Riders",
                     "M Chinnaswamy Stadium, Bengaluru", "Bengaluru", date(2026, 4, 10))


def test_parse_google_news_rss():
    docs = S.GoogleNewsSource.parse_rss(RSS)
    assert len(docs) == 4
    d = docs[0]
    assert d.title == "RCB vs KKR preview: dew expected to play a big role at Chinnaswamy"  # publisher suffix stripped
    assert d.publisher == "ESPNcricinfo" and d.published.startswith("2026-04-08") and not d.full_text


def test_gather_filters_window_relevance_and_dupes(tmp_path):
    class Fake:
        name = "google_news"
        def fetch(self, q):
            return S.GoogleNewsSource.parse_rss(RSS) + S.GoogleNewsSource.parse_rss(RSS)  # duplicates
    docs, notes = S.gather_documents(QUERY, [Fake()], cache_dir=tmp_path)
    titles = [d.title for d in docs]
    assert len(docs) == 2
    assert not any("Stock market" in t for t in titles)          # off-topic dropped
    assert not any("2019" in t for t in titles)                  # outside the date window dropped

    class Boom:
        name = "google_news"
        def fetch(self, q):
            raise AssertionError("should have used cache")
    docs2, notes2 = S.gather_documents(QUERY, [Boom()], cache_dir=tmp_path)
    assert len(docs2) == 2 and any("cached" in n for n in notes2)


def test_parse_gnews_strips_truncation_marker():
    docs = S.GNewsSource.parse(GNEWS)
    assert docs[0].publisher == "Example Sports"
    assert "[2311 chars]" not in docs[0].text and "short boundaries" in docs[0].text


def test_extract_article_prefers_ld_json_and_skips_nav():
    title, body = S.extract_article_text(ARTICLE_HTML)
    assert title == "Preview | Site"
    assert body.startswith("The pitch at the Chinnaswamy") and "Menu item" not in body


def test_url_source_handles_failures(monkeypatch):
    monkeypatch.setattr(S, "fetch_url_text", lambda u, timeout=15: None)
    assert S.URLSource(["https://blocked.example"]).fetch(QUERY) == []


def test_chunking_no_overlap_only_stub_chunk():
    # The original notebook loop produced a final chunk made entirely of overlap.
    words = " ".join(f"w{i}" for i in range(620))
    chunks = chunk_text(words, 300, 50, 50)
    assert len(chunks) == 3 and chunks[-1].split()[-1] == "w619"
    assert len(chunks[-1].split()) == 120
    exact = " ".join(f"w{i}" for i in range(550))  # second chunk ends exactly at the end
    assert len(chunk_text(exact, 300, 50, 50)) == 2
    assert chunk_text("short text", 300, 50, 50) == ["short text"]


def test_chunking_merges_small_tail():
    words = " ".join(f"w{i}" for i in range(320))
    chunks = chunk_text(words, 300, 10, 50)       # tail of 30 new words < 50 -> merged
    assert len(chunks) == 1 and chunks[0].split() == [f"w{i}" for i in range(320)]  # no duplicates


class FakeGeminiClient:
    """Mimics google-genai: models.embed_content(...).embeddings[0].values"""
    def __init__(self, fail_with=None):
        self.calls = 0
        self.fail_with = fail_with
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls += 1
        if self.fail_with:
            raise Exception(self.fail_with)
        rng = np.random.default_rng(abs(hash(contents)) % (2**32))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=rng.normal(size=16).tolist())])


def test_gemini_embeddings_are_cached(tmp_path):
    client = FakeGeminiClient()
    emb = GeminiEmbedder(None, "gemini-embedding-2", EmbeddingCache(tmp_path / "e.sqlite"), client=client)
    a = emb.embed(["one", "two"])
    b = emb.embed(["one", "two"])
    assert a.shape == (2, 16) and np.allclose(a, b) and client.calls == 2   # second pass fully cached


def test_index_falls_back_to_tfidf_on_quota(tmp_path):
    client = FakeGeminiClient(fail_with="429 RESOURCE_EXHAUSTED quota")
    emb = GeminiEmbedder(None, "m", EmbeddingCache(tmp_path / "e.sqlite"), client=client)
    docs = [S.Document("Heavy dew expected at Chinnaswamy tonight, helping the chasing side.", "manual"),
            S.Document("Hazlewood is doubtful with a shoulder niggle.", "manual"),
            S.Document("Ticket prices announced for the stadium.", "manual")]
    idx = VectorIndex(chunk_documents(docs), emb)
    assert idx.backend == "tfidf" and "quota" in idx.fallback_reason
    assert "dew" in idx.search("dew conditions", 1)[0].chunk.text.lower()
    assert "Hazlewood" in idx.search("Hazlewood injury shoulder", 1)[0].chunk.text


def test_tfidf_bigrams():
    t = TfidfEmbedder().fit(["heavy dew tonight", "dew point low heavy rain"])
    m = t.embed(["heavy dew"])
    assert m.shape[0] == 1 and m.sum() > 0


class FakeGroq:
    """Mimics groq SDK: chat.completions.create(...).choices[0].message.content"""
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[
            SimpleNamespace(id="whisper-large-v3"), SimpleNamespace(id="openai/gpt-oss-20b"),
            SimpleNamespace(id="meta-llama/llama-prompt-guard-2-86m")]))
        self.seen = []

    def _create(self, **kw):
        self.seen.append(kw["model"])
        out = self.behaviour(kw["model"])
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out), finish_reason="stop")])


def test_groq_discovers_model_when_configured_one_is_gone():
    def behaviour(model):
        if model == "openai/gpt-oss-20b":
            return "It works [1]."
        return Exception("Error code: 404 - model_not_found")
    g = GroqGenerator(None, ("openai/gpt-oss-120b",), client=FakeGroq(behaviour))
    res = g.generate("hi")
    assert res.model == "openai/gpt-oss-20b" and res.text == "It works [1]."
    assert "whisper-large-v3" not in g._get_client().seen   # non-chat models never tried


def test_groq_rate_limit_raises_and_router_falls_back():
    groq = GroqGenerator(None, ("m",), client=FakeGroq(lambda m: Exception("429 rate_limit_exceeded")))
    with pytest.raises(RateLimited):
        groq.generate("x")

    class FakeGeminiGen:
        provider, available = "gemini", True

        def generate(self, prompt, max_tokens=1400):
            from chimera.rag.llm import LLMResult
            return LLMResult("From Gemini.", "gemini", "gemini-flash-latest")
    res = LLMRouter([groq, FakeGeminiGen()]).generate("x")
    assert res.provider == "gemini"


def test_router_with_no_providers_raises():
    with pytest.raises(LLMError):
        LLMRouter([]).generate("x")


def _team():
    rows = [("V Kohli", "Royal Challengers Bengaluru", "batter", 9.4, 50.0, 2),
            ("SP Narine", "Kolkata Knight Riders", "allrounder", 9.0, 45.0, 1),
            ("JR Hazlewood", "Royal Challengers Bengaluru", "bowler", 8.6, 20.0, 0)]
    df = pd.DataFrame(rows, columns=["player", "team", "role", "credit_value", "ensemble_pred", "batting_position"])
    df["rolling_avg_fantasy_5"], df["matches_played"] = 30.0, 100
    df["captain"] = [True, False, False]
    df["vice_captain"] = [False, True, False]
    df["multiplier"] = [2.0, 1.5, 1.0]
    df["final_points"] = df["ensemble_pred"] * df["multiplier"]
    return df


MATCH = {"team1": "Royal Challengers Bengaluru", "team2": "Kolkata Knight Riders",
         "venue": "M Chinnaswamy Stadium, Bengaluru", "city": "Bengaluru", "date": "2026-04-10",
         "weather": {"weather_temp": 26, "weather_humidity": 70, "weather_dew": 19}, "weather_source": "test"}


def test_explainer_cites_sources_cleans_markdown_and_appends_disclaimer(tmp_path):
    from chimera.config import get_settings

    class LLM:
        available = True

        def __init__(self):
            self.prompt = None

        def generate(self, prompt, max_tokens=1400):
            from chimera.rag.llm import LLMResult
            self.prompt = prompt
            return LLMResult("**V Kohli** leads the XI. Dew should help chasing [1]. Unused ref [9].",
                             "groq", "openai/gpt-oss-120b")
    llm = LLM()
    emb = GeminiEmbedder(None, "m", EmbeddingCache(tmp_path / "e.sqlite"))  # no key -> tfidf
    ex = Explainer(get_settings(), llm=llm, gemini_embedder=emb)
    text = "Heavy dew is expected at the Chinnaswamy tonight. Hazlewood is doubtful with a shoulder niggle. " * 3
    out = ex.explain(MATCH, _team(), RagOptions(use_google_news=False, use_gnews=False, manual_texts=[text]))
    assert out.provider == "groq" and out.context_used and out.retrieval_backend == "tfidf"
    assert "**" not in out.text and out.text.endswith(DISCLAIMER)
    assert [c["id"] for c in out.citations] == [1]            # [9] doesn't exist, so it is ignored
    assert "SP Narine" in llm.prompt and "expected to bat around #2" in llm.prompt
    assert "Do not state bowling styles" in llm.prompt


def test_explainer_without_llm_or_sources_returns_template(tmp_path):
    from chimera.config import get_settings
    emb = GeminiEmbedder(None, "m", EmbeddingCache(tmp_path / "e.sqlite"))
    ex = Explainer(get_settings(), llm=LLMRouter([]), gemini_embedder=emb)
    out = ex.explain(MATCH, _team(), RagOptions(use_google_news=False, use_gnews=False))
    assert out.provider == "template" and not out.context_used
    assert "V Kohli is captain" in out.text
    assert any("No pre-match reporting" in w for w in out.warnings)

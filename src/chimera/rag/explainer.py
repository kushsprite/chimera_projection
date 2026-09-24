"""Explain a selected XI, grounded in live pre-match reporting.

Flow:
    gather documents (pluggable sources) -> chunk -> embed + index
    -> multi-query retrieval -> numbered-source prompt -> LLM -> post-process

Design notes:
  * Player facts (role, projected batting slot, form) are given to the model
    explicitly so it doesn't fill gaps from memory.
  * Sources are numbered and the model is asked to cite them as [n]. We return
    only the citations it actually used.
  * The "projections, not guarantees" caveat is appended by code rather than
    generated, so the model never invents examples to illustrate it.
  * If no LLM is reachable, a deterministic summary is returned instead.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from ..config import Settings
from .chunking import chunk_documents
from .embeddings import EmbeddingCache, GeminiEmbedder
from .llm import GeminiGenerator, GroqGenerator, LLMError, LLMRouter
from .retriever import Hit, VectorIndex
from .sources import (Document, GNewsSource, GoogleNewsSource, ManualSource, MatchQuery, URLSource,
                      gather_documents)

log = logging.getLogger(__name__)

DISCLAIMER = (
    "These picks come from a model trained on historical IPL data. They are projections, "
    "not guarantees, and cannot account for anything not in the data or the sources above, "
    "such as a late change to the playing XI."
)


@dataclass
class RagOptions:
    use_google_news: bool = True
    use_gnews: bool = True
    urls: list[str] = field(default_factory=list)
    manual_texts: list[str] = field(default_factory=list)
    top_k: int = 6
    lookback_days: int = 7
    fetch_full_text: bool = True


@dataclass
class Explanation:
    text: str
    provider: str
    model: Optional[str]
    context_used: bool
    retrieval_backend: str
    citations: list[dict]
    documents_found: int
    chunks_indexed: int
    source_notes: list[str]
    warnings: list[str]


def _surname(name: str) -> str:
    return name.split()[-1]


def _clean(text: str) -> str:
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)      # headers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # bold
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", text)  # italics
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)          # bullets
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class Explainer:
    def __init__(self, settings: Settings, llm: Optional[LLMRouter] = None,
                 gemini_embedder: Optional[GeminiEmbedder] = None):
        self.settings = settings
        cache = EmbeddingCache(settings.paths.cache / "embeddings.sqlite")
        self.embedder = gemini_embedder or GeminiEmbedder(
            settings.gemini_api_key, settings.gemini_embedding_model, cache)
        self.llm = llm or LLMRouter([
            GroqGenerator(settings.groq_api_key, settings.groq_models),
            GeminiGenerator(settings.gemini_api_key, settings.gemini_generation_model),
        ])

    # --------------------------------------------------------------- sources

    def _sources(self, opts: RagOptions) -> list:
        s = self.settings
        sources: list = []
        if opts.manual_texts:
            sources.append(ManualSource(opts.manual_texts))
        if opts.urls:
            sources.append(URLSource(opts.urls))
        if opts.use_gnews and s.gnews_api_key:
            sources.append(GNewsSource(s.gnews_api_key, fetch_full_text=opts.fetch_full_text,
                                       max_fulltext=s.rag_max_fulltext_fetches))
        if opts.use_google_news:
            sources.append(GoogleNewsSource(fetch_full_text=opts.fetch_full_text,
                                            max_fulltext=s.rag_max_fulltext_fetches))
        return sources

    def retrieve(self, match: dict, team: pd.DataFrame, opts: RagOptions) -> tuple[list[Hit], dict]:
        s = self.settings
        q = MatchQuery(team1=match["team1"], team2=match["team2"], venue=match["venue"],
                       city=match.get("city"), match_date=date.fromisoformat(match["date"]),
                       lookback_days=opts.lookback_days)
        docs, notes = gather_documents(q, self._sources(opts), cache_dir=s.paths.cache,
                                       cache_ttl_hours=s.rag_doc_cache_ttl_hours)
        chunks = chunk_documents(docs, s.rag_chunk_words, s.rag_chunk_overlap, s.rag_min_chunk_words)
        index = VectorIndex(chunks, self.embedder)

        venue_short = match["venue"].split(",")[0]
        top = team.sort_values("ensemble_pred", ascending=False)["player"].head(3).tolist()
        queries = [
            f"{venue_short} pitch conditions surface batting bowling expected score",
            "weather forecast dew rain humidity evening",
            f"{q.short1} {q.short2} team news injury ruled out playing XI changes",
            f"{q.short1} vs {q.short2} key players form preview",
        ] + [f"{_surname(p)} form" for p in top]
        hits = index.multi_search(queries, per_query=3, total=opts.top_k)
        meta = {"documents_found": len(docs), "chunks_indexed": len(chunks), "backend": index.backend,
                "fallback_reason": index.fallback_reason, "notes": notes}
        return hits, meta

    # ---------------------------------------------------------------- prompt

    @staticmethod
    def build_prompt(match: dict, team: pd.DataFrame, hits: list[Hit]) -> str:
        lines = []
        for _, r in team.iterrows():
            tag = " | CAPTAIN (2x)" if r.get("captain") else (" | VICE-CAPTAIN (1.5x)" if r.get("vice_captain") else "")
            bat = int(r["batting_position"]) if pd.notna(r.get("batting_position")) else 0
            bat_txt = "not expected to bat" if bat == 0 else f"expected to bat around #{bat}"
            lines.append(
                f"- {r['player']} ({r['team']}, {r['role']}){tag}: projected {r['ensemble_pred']:.1f} pts, "
                f"{r['credit_value']:.1f} credits, {bat_txt}, last-5 avg {r.get('rolling_avg_fantasy_5', 0):.1f} pts, "
                f"{int(r.get('matches_played', 0))} IPL matches"
            )
        team_block = "\n".join(lines)
        credits = team["credit_value"].sum()
        total = team["final_points"].sum()
        wx = match.get("weather", {})
        wx_line = (f"Model weather input for {match.get('city')}: {wx.get('weather_temp', 0):.0f}C, "
                   f"{wx.get('weather_humidity', 0):.0f}% humidity, dew point {wx.get('weather_dew', 0):.0f}C "
                   f"(source: {match.get('weather_source')}).") if wx else ""

        if hits:
            ctx = "\n\n".join(
                f"[{i + 1}] {h.chunk.doc.title or 'Untitled'} ({h.chunk.doc.publisher or h.chunk.doc.source}"
                f"{', ' + h.chunk.doc.published[:10] if h.chunk.doc.published else ''}"
                f"{', headline only' if not h.chunk.doc.full_text else ''})\n{h.chunk.text}"
                for i, h in enumerate(hits)
            )
            context_rules = (
                "Use the numbered SOURCES for anything about pitch, weather, injuries, or team news, and cite "
                "them inline as [1], [2]. Headline-only sources are thin, so treat them cautiously."
            )
        else:
            ctx = "No current reporting was found for this match."
            context_rules = ("No sources are available. Say briefly that no current pre-match reporting was "
                             "found, and reason only from the team data.")

        return f"""You are a cricket fantasy analyst. Explain why this model-selected fantasy XI makes sense for {match['team1']} vs {match['team2']} at {match['venue']} on {match['date']}.

SOURCES (live pre-match reporting):
{ctx}

SELECTED XI ({credits:.1f}/100 credits, {total:.1f} projected points with captain multipliers):
{team_block}

{wx_line}

Write 180-260 words in plain paragraphs covering:
1. Why the captain and vice-captain were chosen.
2. The batting and bowling balance of the XI.
3. Anything in the sources that supports or cuts against these picks (pitch, dew, injuries, form). Call out any tension between the model's numbers and the reporting.

Rules:
- {context_rules}
- Only discuss players in the SELECTED XI. Refer to them exactly as written (e.g. "SP Narine"); do not expand or guess full names.
- Do not state bowling styles, batting positions, or records unless they appear above or in the sources.
- If something is uncertain, leave it out rather than guess.
- No markdown, no bullet points, no headings, no closing disclaimer."""

    # ---------------------------------------------------------------- output

    @staticmethod
    def _template(match: dict, team: pd.DataFrame, hits: list[Hit]) -> str:
        cap = team[team["captain"]].iloc[0]
        vc = team[team["vice_captain"]].iloc[0]
        roles = team["role"].value_counts().to_dict()
        mix = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in roles.items())
        by_team = team["team"].value_counts().to_dict()
        split = " and ".join(f"{v} from {k}" for k, v in by_team.items())
        text = (
            f"{cap['player']} is captain as the highest projected scorer ({cap['ensemble_pred']:.1f} pts), "
            f"with {vc['player']} as vice-captain ({vc['ensemble_pred']:.1f} pts). The XI is {mix}, "
            f"{split}, using {team['credit_value'].sum():.1f} of 100 credits for "
            f"{team['final_points'].sum():.1f} projected points including multipliers."
        )
        if hits:
            heads = "; ".join(f"[{i + 1}] {h.chunk.doc.title}" for i, h in enumerate(hits[:4]) if h.chunk.doc.title)
            text += f" Relevant pre-match reporting: {heads}."
        return text

    def explain(self, match: dict, team: pd.DataFrame, opts: Optional[RagOptions] = None) -> Explanation:
        opts = opts or RagOptions()
        warnings: list[str] = []
        hits, meta = self.retrieve(match, team, opts)
        if meta["fallback_reason"] and meta["chunks_indexed"]:
            warnings.append(f"Retrieval used TF-IDF instead of Gemini embeddings: {meta['fallback_reason']}")
        if not hits:
            warnings.append("No pre-match reporting found; explanation is based on model data only.")

        prompt = self.build_prompt(match, team, hits)
        provider, model = "template", None
        try:
            res = self.llm.generate(prompt)
            text, provider, model = _clean(res.text), res.provider, res.model
            if res.truncated:
                warnings.append("LLM output hit its length limit and may be cut short.")
        except LLMError as e:
            warnings.append(f"LLM unavailable, returned a template summary instead: {e}")
            text = self._template(match, team, hits)

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        citations = []
        for n in cited:
            d: Document = hits[n - 1].chunk.doc
            citations.append({"id": n, "title": d.title, "url": d.url, "publisher": d.publisher,
                              "published": d.published, "source": d.source, "full_text": d.full_text,
                              "score": round(hits[n - 1].score, 3)})

        return Explanation(
            text=f"{text}\n\n{DISCLAIMER}", provider=provider, model=model, context_used=bool(hits),
            retrieval_backend=meta["backend"], citations=citations, documents_found=meta["documents_found"],
            chunks_indexed=meta["chunks_indexed"], source_notes=meta["notes"], warnings=warnings,
        )

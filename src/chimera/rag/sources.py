"""Where RAG documents come from.

Every source implements the same contract, `fetch(query) -> list[Document]`,
so the rest of the pipeline (chunking, embedding, retrieval, prompting) never
knows or cares where the text came from. Add a new source by writing one class.

Sources:
    ManualSource       text you paste in
    URLSource          article URLs you supply (full text extracted from HTML)
    GoogleNewsSource   Google News RSS search, free, no key (headlines + summaries)
    GNewsSource        gnews.io API, free tier with GNEWS_API_KEY (real article URLs,
                       so full text can be fetched)
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Protocol
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

from ..constants import TEAM_SHORT

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_HTML_BYTES = 3_000_000


# ------------------------------------------------------------------ data types

@dataclass
class Document:
    text: str
    source: str                      # manual | url | google_news | gnews
    title: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    published: Optional[str] = None  # ISO 8601
    full_text: bool = False          # False when we only have a headline/summary

    @property
    def doc_id(self) -> str:
        basis = self.url or (self.title or "") + self.text[:500]
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchQuery:
    team1: str
    team2: str
    venue: str
    city: Optional[str]
    match_date: date
    lookback_days: int = 7

    @property
    def short1(self) -> str:
        return TEAM_SHORT.get(self.team1, self.team1)

    @property
    def short2(self) -> str:
        return TEAM_SHORT.get(self.team2, self.team2)

    @property
    def window(self) -> tuple[date, date]:
        return self.match_date - timedelta(days=self.lookback_days), self.match_date + timedelta(days=1)

    def search_queries(self) -> list[str]:
        return [
            f"{self.short1} vs {self.short2} preview",
            f"{self.team1} {self.team2} pitch report",
            f"{self.short1} {self.short2} team news injury",
        ]

    def relevance_terms(self) -> list[str]:
        terms = {self.team1, self.team2, self.short1, self.short2,
                 self.team1.split()[-1], self.team2.split()[-1]}
        if self.city:
            terms.add(self.city)
        return [t.lower() for t in terms if t and len(t) >= 2]


class DocumentSource(Protocol):
    name: str

    def fetch(self, query: MatchQuery) -> list[Document]: ...


# -------------------------------------------------------------- HTML -> text

class _ArticleText(HTMLParser):
    """Stdlib-only paragraph extractor. Keeps <p> text outside boilerplate tags."""

    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth_skip = 0
        self.in_p = False
        self.buf: list[str] = []
        self.paragraphs: list[str] = []
        self.ld_json: list[str] = []
        self._in_ld = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld = True
            return
        if tag in self.SKIP:
            self.depth_skip += 1
        elif tag == "p" and not self.depth_skip:
            self.in_p, self.buf = True, []

    def handle_endtag(self, tag):
        if tag == "script" and self._in_ld:
            self._in_ld = False
            return
        if tag in self.SKIP and self.depth_skip:
            self.depth_skip -= 1
        elif tag == "p" and self.in_p:
            text = " ".join("".join(self.buf).split())
            if len(text) >= 40:
                self.paragraphs.append(text)
            self.in_p = False

    def handle_data(self, data):
        if self._in_ld:
            self.ld_json.append(data)
        elif self.in_p and not self.depth_skip:
            self.buf.append(data)


def _article_body_from_ld_json(chunks: list[str]) -> Optional[str]:
    for raw in chunks:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                graph = item.get("@graph")
                for node in ([item] + (graph if isinstance(graph, list) else [])):
                    if isinstance(node, dict) and isinstance(node.get("articleBody"), str):
                        body = " ".join(node["articleBody"].split())
                        if len(body) > 200:
                            return body
    return None


def extract_article_text(page_html: str) -> tuple[Optional[str], Optional[str]]:
    """Return (title, body). Uses trafilatura if installed, else a stdlib parser."""
    try:
        import trafilatura  # optional, better quality
        body = trafilatura.extract(page_html, include_comments=False, include_tables=False)
        meta = trafilatura.extract_metadata(page_html)
        if body and len(body) > 200:
            return (meta.title if meta else None), body
    except ImportError:
        pass
    except Exception as e:  # trafilatura can fail on odd pages
        log.debug("trafilatura failed: %s", e)

    parser = _ArticleText()
    try:
        parser.feed(page_html)
    except Exception as e:
        log.debug("HTML parse error: %s", e)
    m = re.search(r"<title[^>]*>(.*?)</title>", page_html, re.I | re.S)
    title = html.unescape(" ".join(m.group(1).split())) if m else None
    body = _article_body_from_ld_json(parser.ld_json) or "\n\n".join(parser.paragraphs)
    return title, (body if len(body) > 200 else None)


def fetch_url_text(url: str, timeout: int = 15) -> Optional[Document]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
                         timeout=timeout, stream=True)
        if r.status_code != 200:
            log.info("URL %s returned %s", url, r.status_code)
            return None
        content = r.raw.read(MAX_HTML_BYTES, decode_content=True)
        page = content.decode(r.encoding or "utf-8", errors="replace")
    except Exception as e:
        log.info("URL fetch failed for %s: %s", url, e)
        return None
    title, body = extract_article_text(page)
    if not body:
        return None
    return Document(text=body, source="url", title=title, url=r.url, full_text=True)


# ------------------------------------------------------------------- sources

class ManualSource:
    name = "manual"

    def __init__(self, texts: list[str]):
        self.texts = [t for t in texts if t and t.strip()]

    def fetch(self, query: MatchQuery) -> list[Document]:
        return [Document(text=t.strip(), source="manual", title=f"Provided text {i + 1}", full_text=True)
                for i, t in enumerate(self.texts)]


class URLSource:
    name = "url"

    def __init__(self, urls: list[str]):
        self.urls = [u for u in urls if u and u.strip()]

    def fetch(self, query: MatchQuery) -> list[Document]:
        docs = []
        for u in self.urls:
            d = fetch_url_text(u.strip())
            if d:
                docs.append(d)
            else:
                log.warning("Could not extract article text from %s", u)
        return docs


def _strip_tags(s: str) -> str:
    return html.unescape(" ".join(re.sub(r"<[^>]+>", " ", s or "").split()))


class GoogleNewsSource:
    """Google News RSS search. Free and keyless.

    Google News links are redirect URLs that usually need JavaScript to resolve,
    so by default documents are headline + summary only (still useful: "X ruled
    out", "pitch expected to favour spin"). If the optional `googlenewsdecoder`
    package is installed, links are decoded and full articles fetched.
    """

    name = "google_news"
    BASE = "https://news.google.com/rss/search"

    def __init__(self, max_items_per_query: int = 8, fetch_full_text: bool = True, max_fulltext: int = 4):
        self.max_items = max_items_per_query
        self.fetch_full_text = fetch_full_text
        self.max_fulltext = max_fulltext

    def _url(self, q: str, query: MatchQuery) -> str:
        start, end = query.window
        full = f"{q} after:{start.isoformat()} before:{end.isoformat()}"
        return f"{self.BASE}?q={quote_plus(full)}&hl=en-IN&gl=IN&ceid=IN:en"

    @staticmethod
    def parse_rss(xml_text: str) -> list[Document]:
        docs = []
        root = ElementTree.fromstring(xml_text)
        for item in root.iter("item"):
            title = _strip_tags(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            desc = _strip_tags(item.findtext("description") or "")
            src_el = item.find("source")
            publisher = src_el.text.strip() if src_el is not None and src_el.text else None
            published = None
            pub = item.findtext("pubDate")
            if pub:
                try:
                    published = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
                except (TypeError, ValueError):
                    pass
            if publisher and title.endswith(f" - {publisher}"):
                title = title[: -len(publisher) - 3]
            body = title if (not desc or desc.startswith(title)) else f"{title}. {desc}"
            if title:
                docs.append(Document(text=body, source="google_news", title=title, url=link,
                                     publisher=publisher, published=published, full_text=False))
        return docs

    def _decode(self, url: str) -> Optional[str]:
        try:
            from googlenewsdecoder import gnewsdecoder  # optional dependency
        except ImportError:
            return None
        try:
            res = gnewsdecoder(url, interval=1)
            return res.get("decoded_url") if res.get("status") else None
        except Exception as e:
            log.debug("Google News decode failed: %s", e)
            return None

    def fetch(self, query: MatchQuery) -> list[Document]:
        docs: list[Document] = []
        for q in query.search_queries():
            try:
                r = requests.get(self._url(q, query), headers={"User-Agent": USER_AGENT}, timeout=15)
                r.raise_for_status()
                docs.extend(self.parse_rss(r.text)[: self.max_items])
            except Exception as e:
                log.warning("Google News query failed (%s): %s", q, e)
            time.sleep(0.3)

        if self.fetch_full_text:
            upgraded = 0
            for i, d in enumerate(docs):
                if upgraded >= self.max_fulltext or not d.url:
                    continue
                real = self._decode(d.url)
                if not real:
                    continue
                full = fetch_url_text(real)
                if full:
                    docs[i] = Document(text=full.text, source="google_news", title=d.title, url=real,
                                       publisher=d.publisher, published=d.published, full_text=True)
                    upgraded += 1
        return docs


class GNewsSource:
    """gnews.io search API. Needs GNEWS_API_KEY (free tier: 100 requests/day)."""

    name = "gnews"
    BASE = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: str, max_queries: int = 2, fetch_full_text: bool = True, max_fulltext: int = 4):
        self.api_key = api_key
        self.max_queries = max_queries
        self.fetch_full_text = fetch_full_text
        self.max_fulltext = max_fulltext

    @staticmethod
    def parse(payload: dict) -> list[Document]:
        docs = []
        for a in payload.get("articles", []):
            title = (a.get("title") or "").strip()
            parts = [title, (a.get("description") or "").strip(), (a.get("content") or "").strip()]
            # free tier truncates content with "... [1234 chars]"
            text = " ".join(p for p in parts if p)
            text = re.sub(r"\s*\.\.\.\s*\[\d+ chars\]\s*$", "", text)
            if title:
                docs.append(Document(text=text, source="gnews", title=title, url=a.get("url"),
                                     publisher=(a.get("source") or {}).get("name"),
                                     published=a.get("publishedAt"), full_text=False))
        return docs

    def fetch(self, query: MatchQuery) -> list[Document]:
        start, end = query.window
        docs: list[Document] = []
        for q in query.search_queries()[: self.max_queries]:
            params = {"q": q, "lang": "en", "max": 10, "apikey": self.api_key,
                      "from": f"{start.isoformat()}T00:00:00Z", "to": f"{end.isoformat()}T23:59:59Z"}
            try:
                r = requests.get(self.BASE, params=params, timeout=15)
                if r.status_code in (401, 403, 429):
                    log.warning("GNews returned %s (key invalid or daily quota reached)", r.status_code)
                    break
                r.raise_for_status()
                docs.extend(self.parse(r.json()))
            except Exception as e:
                log.warning("GNews query failed (%s): %s", q, e)

        if self.fetch_full_text:
            upgraded = 0
            for i, d in enumerate(docs):
                if upgraded >= self.max_fulltext or not d.url:
                    continue
                full = fetch_url_text(d.url)
                if full:
                    docs[i] = Document(text=full.text, source="gnews", title=d.title, url=d.url,
                                       publisher=d.publisher, published=d.published, full_text=True)
                    upgraded += 1
        return docs


# ------------------------------------------------------------- aggregation

def _norm_title(t: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:80]


def _within_window(d: Document, query: MatchQuery) -> bool:
    if not d.published:
        return True  # manual text and pasted URLs have no date; trust the caller
    try:
        when = datetime.fromisoformat(d.published.replace("Z", "+00:00")).date()
    except ValueError:
        return True
    start, end = query.window
    return start <= when <= end


def _is_relevant(d: Document, query: MatchQuery) -> bool:
    if d.source in ("manual", "url"):
        return True
    hay = f"{d.title or ''} {d.text[:2000]}".lower()
    return any(t in hay for t in query.relevance_terms())


def gather_documents(query: MatchQuery, sources: list, cache_dir: Optional[Path] = None,
                     cache_ttl_hours: float = 3.0) -> tuple[list[Document], list[str]]:
    """Fetch from every source, filter by date/relevance, de-duplicate.

    Network sources are cached on disk per match for `cache_ttl_hours`.
    """
    notes: list[str] = []
    all_docs: list[Document] = []
    for src in sources:
        cache_file = None
        if cache_dir is not None and src.name in ("google_news", "gnews"):
            key = hashlib.sha1(f"{src.name}|{query.team1}|{query.team2}|{query.match_date}".encode()).hexdigest()[:16]
            cache_file = Path(cache_dir) / f"docs_{src.name}_{key}.json"
            if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < cache_ttl_hours * 3600:
                try:
                    cached = [Document(**d) for d in json.loads(cache_file.read_text())]
                    all_docs.extend(cached)
                    notes.append(f"{src.name}: {len(cached)} documents (cached)")
                    continue
                except Exception:
                    pass
        try:
            got = src.fetch(query)
        except Exception as e:
            notes.append(f"{src.name}: failed ({e})")
            continue
        notes.append(f"{src.name}: {len(got)} documents")
        all_docs.extend(got)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps([d.to_dict() for d in got]))

    kept, seen_urls, seen_titles = [], set(), set()
    for d in all_docs:
        if not _within_window(d, query) or not _is_relevant(d, query):
            continue
        t = _norm_title(d.title)
        if (d.url and d.url in seen_urls) or (t and t in seen_titles and d.source not in ("manual",)):
            continue
        if d.url:
            seen_urls.add(d.url)
        if t:
            seen_titles.add(t)
        kept.append(d)

    # full articles first, then newest
    kept.sort(key=lambda d: (d.full_text, d.published or ""), reverse=True)
    dropped = len(all_docs) - len(kept)
    if dropped:
        notes.append(f"dropped {dropped} duplicate, off-topic, or out-of-window documents")
    return kept, notes

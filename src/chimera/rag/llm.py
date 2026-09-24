"""Text generation with provider fallback and model auto-discovery.

Providers deprecate model names often (gemini-2.5-flash and
llama-3.3-70b-versatile both disappeared during development), so instead of
hardcoding one name we try a preference list and, on "model not found", ask
the provider which models exist and pick the best available.

Order: Groq (fast, generous free tier) -> Gemini. If both fail, the caller
falls back to a deterministic template so an answer is always returned.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

NON_CHAT_MARKERS = ("whisper", "guard", "orpheus", "tts", "safeguard", "transcribe", "embedding",
                    "image", "lyria", "robotics", "computer-use", "deep-research", "antigravity", "omni")


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    truncated: bool = False


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "rate_limit" in s or "rate limit" in s or "resource_exhausted" in s or "quota" in s


def _is_not_found(e: Exception) -> bool:
    s = str(e).lower()
    return "404" in s or "not_found" in s or "model_not_found" in s or "does not exist" in s


def _is_transient(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("500", "502", "503", "504", "unavailable", "timeout", "timed out", "overloaded",
                                "connection"))


class GroqGenerator:
    provider = "groq"

    def __init__(self, api_key: Optional[str], preferred: tuple, client=None):
        self.api_key = api_key
        self.preferred = list(preferred)
        self._client = client
        self._working_model: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _discover(self) -> list[str]:
        ids = [m.id for m in self._get_client().models.list().data]
        chat = [i for i in ids if not any(k in i.lower() for k in NON_CHAT_MARKERS)]
        return [p for p in self.preferred if p in chat] + [c for c in chat if c not in self.preferred]

    def _call(self, model: str, prompt: str, max_tokens: int) -> LLMResult:
        client = self._get_client()
        kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}],
                      max_tokens=max_tokens, temperature=0.3)
        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"  # keeps reasoning tokens from eating the answer budget
        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMError(f"{model} returned an empty response")
        return LLMResult(text, self.provider, model, truncated=(choice.finish_reason == "length"))

    def generate(self, prompt: str, max_tokens: int = 1400) -> LLMResult:
        candidates = [self._working_model] if self._working_model else list(self.preferred)
        tried: set[str] = set()
        discovered = False
        errors = []
        while candidates:
            model = candidates.pop(0)
            if model in tried:
                continue
            tried.add(model)
            for attempt in range(3):
                try:
                    res = self._call(model, prompt, max_tokens)
                    self._working_model = model
                    return res
                except Exception as e:
                    if _is_rate_limit(e):
                        raise RateLimited(f"Groq rate limit: {e}") from e
                    if _is_not_found(e) or "decommission" in str(e).lower():
                        errors.append(f"{model}: not available")
                        break
                    if _is_transient(e) and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    errors.append(f"{model}: {e}")
                    break
            if not candidates and not discovered:
                discovered = True
                try:
                    candidates = [m for m in self._discover() if m not in tried]
                except Exception as e:
                    errors.append(f"model discovery failed: {e}")
        raise LLMError("Groq failed: " + "; ".join(errors[-4:]))


class GeminiGenerator:
    provider = "gemini"

    def __init__(self, api_key: Optional[str], model: str, client=None):
        self.api_key = api_key
        self.model = model
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1400) -> LLMResult:
        from google.genai import types
        client = self._get_client()
        cfg = types.GenerateContentConfig(temperature=0.3, max_output_tokens=max_tokens)
        last = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                text = (resp.text or "").strip()
                if not text:
                    raise LLMError("Gemini returned an empty response")
                finish = ""
                try:
                    finish = str(resp.candidates[0].finish_reason)
                except Exception:
                    pass
                return LLMResult(text, self.provider, self.model, truncated="MAX_TOKENS" in finish)
            except Exception as e:
                last = e
                if _is_rate_limit(e):
                    raise RateLimited(f"Gemini quota: {e}") from e
                if _is_transient(e) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break
        raise LLMError(f"Gemini failed: {last}")


class LLMRouter:
    def __init__(self, generators: list):
        self.generators = [g for g in generators if g.available]

    @property
    def available(self) -> bool:
        return bool(self.generators)

    def generate(self, prompt: str, max_tokens: int = 1400) -> LLMResult:
        errors = []
        for g in self.generators:
            try:
                res = g.generate(prompt, max_tokens)
                if res.truncated:
                    log.info("%s output hit the token limit, retrying with more room", g.provider)
                    try:
                        bigger = g.generate(prompt, max_tokens * 2)
                        if not bigger.truncated or len(bigger.text) > len(res.text):
                            res = bigger
                    except Exception:
                        pass
                return res
            except Exception as e:
                errors.append(f"{g.provider}: {e}")
                log.warning("LLM provider %s failed: %s", g.provider, e)
        raise LLMError("All LLM providers failed. " + " | ".join(errors) if errors
                       else "No LLM provider configured (set GROQ_API_KEY or GEMINI_API_KEY)")

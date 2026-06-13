"""OpenAI-compatible LLM client with provider fallback.

Provider order comes from RunConfig.llm_providers. The first provider with
an API key in the environment is used; on rate-limit (429) or server error,
the next provider is tried.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

from ..config import ProviderConfig


# How many times to wait-and-retry a single provider on a 429 before moving on.
_MAX_RETRIES_PER_PROVIDER = 3
# Cap on how long we'll honor a server-suggested retry delay (seconds).
_MAX_RETRY_WAIT = 65.0


class LLMClient:
    def __init__(self, providers: list[ProviderConfig]):
        self._providers = providers
        # Per-provider timestamp of the last call, for proactive rpm spacing.
        self._last_call: dict[str, float] = {}
        # Provider/model of the most recent successful call (for reporting).
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None

    def complete(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> tuple[dict, str, int, int]:
        """Return (parsed_json, model_used, input_tokens, output_tokens).

        On a 429 the same provider is retried after the server-suggested delay
        (up to a cap) before falling through to the next provider. Calls are also
        proactively spaced to respect each provider's rpm_limit.

        Raises RuntimeError if all providers fail or are unavailable.
        """
        last_error: Optional[Exception] = None

        for provider in self._providers:
            if not provider.available:
                continue

            for attempt in range(_MAX_RETRIES_PER_PROVIDER):
                self._respect_rpm(provider)
                try:
                    return self._call(provider, messages, response_format, max_tokens, temperature)
                except RateLimitError as e:
                    last_error = e
                    # A per-DAY quota won't recover by waiting seconds — don't spin.
                    if _is_daily_quota(e):
                        print(f"  [llm] {provider.name} daily free-tier quota exhausted; "
                              f"skipping retries.")
                        break
                    wait = _retry_delay_seconds(e)
                    if attempt < _MAX_RETRIES_PER_PROVIDER - 1:
                        print(f"  [llm] {provider.name} rate-limited; waiting {wait:.0f}s "
                              f"(attempt {attempt + 1}/{_MAX_RETRIES_PER_PROVIDER})...")
                        time.sleep(wait)
                        continue
                    break  # exhausted retries on this provider → try next provider
                except APIConnectionError as e:
                    last_error = e
                    # Provider unreachable (e.g. local Ollama not running) → try next.
                    print(f"  [llm] {provider.name} unreachable; falling through.")
                    break
                except APIStatusError as e:
                    last_error = e
                    break  # non-429 API error → try next provider
                except Exception as e:
                    last_error = e
                    return self._reraise(last_error)  # client-side bug, don't mask it

        raise RuntimeError(
            f"All LLM providers failed or unavailable. Last error: {last_error}"
        )

    @staticmethod
    def _reraise(err: Exception):
        raise RuntimeError(f"All LLM providers failed or unavailable. Last error: {err}")

    def _respect_rpm(self, provider: ProviderConfig) -> None:
        """Sleep just enough to keep under the provider's requests-per-minute limit."""
        rpm = max(provider.rpm_limit, 1)
        min_interval = 60.0 / rpm
        last = self._last_call.get(provider.name)
        if last is not None:
            elapsed = time.time() - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_call[provider.name] = time.time()

    def _call(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        response_format: Optional[dict],
        max_tokens: int,
        temperature: float,
    ) -> tuple[dict, str, int, int]:
        client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
        )

        kwargs: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""

        # Parse JSON — retry once if the model wrapped it in markdown fences
        parsed = _parse_json(content)
        if parsed is None:
            # strip markdown fences and retry
            cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = _parse_json(cleaned)
        if parsed is None:
            raise ValueError(f"LLM returned non-JSON: {content[:200]}")

        input_tokens = resp.usage.prompt_tokens if resp.usage else 0
        output_tokens = resp.usage.completion_tokens if resp.usage else 0
        self.last_provider = provider.name
        self.last_model = provider.model
        return parsed, provider.model, input_tokens, output_tokens


def _is_daily_quota(err: Exception) -> bool:
    """True if the 429 is a per-day quota violation (won't clear by waiting seconds).

    Keyed on the quotaId, which Google labels with 'PerDay' vs 'PerMinute'.
    """
    text = str(err)
    return "PerDay" in text or "RequestsPerDay" in text


def _retry_delay_seconds(err: Exception) -> float:
    """Extract the server-suggested retry delay from a 429, else exponential-ish default.

    Google returns a RetryInfo block like 'Please retry in 6.22s' / "retryDelay": "6s".
    We honor it (capped) so we wait exactly as long as needed and no longer.
    """
    text = str(err)
    # e.g. "retry in 6.220328526s" or "'retryDelay': '6s'"
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", text) or re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", text)
    if m:
        return min(float(m.group(1)) + 1.0, _MAX_RETRY_WAIT)  # +1s safety margin
    return 30.0  # no hint provided — conservative default


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def prompt_hash(messages: list[dict]) -> str:
    raw = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

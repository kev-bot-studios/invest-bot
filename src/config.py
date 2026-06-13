"""Run configuration loaded from YAML with env-var key resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=value pairs from a .env file into os.environ.

    Minimal, dependency-free. Existing environment variables take precedence,
    so an explicit `export` always overrides the file. Lines starting with #
    and blank lines are ignored; surrounding quotes on values are stripped.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: Optional[str]  # None/empty → keyless local provider (e.g. Ollama)
    model: str
    rpm_limit: int = 10

    @property
    def api_key(self) -> Optional[str]:
        # Keyless providers (local Ollama) still need a non-empty string because the
        # OpenAI client requires one; the value is ignored by the local server.
        if not self.api_key_env:
            return "local"
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        # A keyless provider is assumed reachable; if the local server is down the
        # call fails fast and the client falls through to the next provider.
        if not self.api_key_env:
            return True
        return bool(os.environ.get(self.api_key_env))


@dataclass
class SourcesConfig:
    yfinance_prices: bool = True
    yfinance_fundamentals: bool = True
    yfinance_news: bool = True
    edgar_companyfacts: bool = True
    edgar_filings: bool = True
    edgar_form4: bool = True
    fred: bool = True
    finnhub: bool = False


@dataclass
class WeightsConfig:
    valuation: float = 0.20
    quality: float = 0.25
    growth: float = 0.25
    momentum: float = 0.15
    insider: float = 0.05
    filing_tone: float = 0.05
    news_tone: float = 0.05


@dataclass
class RunConfig:
    tickers: list[str]
    sector: Optional[str]  # required at run time; validated against available_sectors()
    sources: SourcesConfig
    llm_providers: list[ProviderConfig]
    weights: WeightsConfig
    db_path: str = "invest_bot.db"
    cache_dir: str = "cache"
    prompt_version: str = "v1"

    def first_available_provider(self) -> Optional[ProviderConfig]:
        for p in self.llm_providers:
            if p.available:
                return p
        return None


def load_config(path: str | Path = "config/run_config.yaml") -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    sources_raw = raw.get("sources", {})
    sources = SourcesConfig(
        yfinance_prices=sources_raw.get("yfinance_prices", True),
        yfinance_fundamentals=sources_raw.get("yfinance_fundamentals", True),
        yfinance_news=sources_raw.get("yfinance_news", True),
        edgar_companyfacts=sources_raw.get("edgar_companyfacts", True),
        edgar_filings=sources_raw.get("edgar_filings", True),
        edgar_form4=sources_raw.get("edgar_form4", True),
        fred=sources_raw.get("fred", True),
        finnhub=sources_raw.get("finnhub", False),
    )

    providers = [
        ProviderConfig(
            name=p["name"],
            base_url=p["base_url"],
            api_key_env=p.get("api_key_env"),
            model=p["model"],
            rpm_limit=p.get("rpm_limit", 10),
        )
        for p in raw.get("llm_providers", [])
    ]

    weights_raw = raw.get("weights", {})
    weights = WeightsConfig(
        valuation=weights_raw.get("valuation", 0.20),
        quality=weights_raw.get("quality", 0.25),
        growth=weights_raw.get("growth", 0.25),
        momentum=weights_raw.get("momentum", 0.15),
        insider=weights_raw.get("insider", 0.05),
        filing_tone=weights_raw.get("filing_tone", 0.05),
        news_tone=weights_raw.get("news_tone", 0.05),
    )

    return RunConfig(
        tickers=[t.upper() for t in raw["tickers"]],
        sector=raw.get("sector"),  # optional here; required/validated by the caller
        sources=sources,
        llm_providers=providers,
        weights=weights,
        db_path=raw.get("db_path", "invest_bot.db"),
        cache_dir=raw.get("cache_dir", "cache"),
        prompt_version=raw.get("prompt_version", "v1"),
    )


def available_sectors(peers_dir: str | Path = "config/peers") -> list[str]:
    """Sector names with a peer-universe file in peers_dir (the `<name>` of <name>.yaml)."""
    d = Path(peers_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_peer_tickers(sector: str, peers_dir: str | Path = "config/peers") -> list[str]:
    path = Path(peers_dir) / f"{sector}.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [t.upper() for t in raw.get("tickers", [])]

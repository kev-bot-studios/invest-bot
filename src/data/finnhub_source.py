"""Finnhub data source (optional). Requires FINNHUB_API_KEY environment variable."""

from __future__ import annotations

import os
from typing import Optional

import requests

from .cache import Cache


_FINNHUB_TTL_H = 24 * 7
_BASE = "https://finnhub.io/api/v1"


def fetch_analyst_blob(ticker: str, cache: Cache) -> dict:
    """Fetch analyst recommendations + earnings surprises from Finnhub."""
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return {"unavailable": True, "reason": "FINNHUB_API_KEY not set"}

    key = f"finnhub_{ticker}"
    cached = cache.get(key, _FINNHUB_TTL_H)
    if cached is not None:
        return cached

    headers = {"X-Finnhub-Token": api_key}
    result: dict = {"ticker": ticker}

    # Analyst recommendations (latest)
    try:
        resp = requests.get(
            f"{_BASE}/stock/recommendation",
            params={"symbol": ticker},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        recs = resp.json()
        if recs:
            latest = recs[0]
            result["recommendations"] = {
                "period": latest.get("period"),
                "strong_buy": latest.get("strongBuy"),
                "buy": latest.get("buy"),
                "hold": latest.get("hold"),
                "sell": latest.get("sell"),
                "strong_sell": latest.get("strongSell"),
            }
    except Exception:
        result["recommendations"] = None

    # Earnings surprises (last 4 quarters)
    try:
        resp = requests.get(
            f"{_BASE}/stock/earnings",
            params={"symbol": ticker, "limit": 4},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        earnings = resp.json()
        result["earnings_surprises"] = [
            {
                "period": e.get("period"),
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "surprise_pct": e.get("surprisePercent"),
            }
            for e in earnings
        ]
    except Exception:
        result["earnings_surprises"] = None

    # Price target
    try:
        resp = requests.get(
            f"{_BASE}/stock/price-target",
            params={"symbol": ticker},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        pt = resp.json()
        result["price_target"] = {
            "mean": pt.get("targetMean"),
            "high": pt.get("targetHigh"),
            "low": pt.get("targetLow"),
            "analyst_count": pt.get("numberOfAnalysts"),
        }
    except Exception:
        result["price_target"] = None

    cache.set(key, result)
    return result

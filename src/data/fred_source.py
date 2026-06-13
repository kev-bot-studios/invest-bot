"""FRED macro data source. Requires FRED_API_KEY environment variable."""

from __future__ import annotations

import os
from typing import Optional

import requests

from .cache import Cache


_FRED_TTL_H = 24 * 7
_BASE = "https://api.stlouisfed.org/fred"

# Sector → relevant FRED series IDs
SECTOR_SERIES: dict[str, list[tuple[str, str]]] = {
    "technology": [
        ("DFF", "Fed Funds Rate"),
        ("T10Y2Y", "10Y-2Y Yield Spread"),
        ("NASDAQCOM", "NASDAQ Composite"),
        ("PCE", "Personal Consumption Expenditures"),
    ],
    "energy": [
        ("DFF", "Fed Funds Rate"),
        ("DCOILWTICO", "WTI Crude Oil Price"),
        ("GASREGCOVW", "US Regular Gasoline Price"),
        ("T10Y2Y", "10Y-2Y Yield Spread"),
    ],
    "financials": [
        ("DFF", "Fed Funds Rate"),
        ("T10Y2Y", "10Y-2Y Yield Spread"),
        ("MORTGAGE30US", "30-Year Mortgage Rate"),
        ("BAA10Y", "Baa Corporate Spread"),
    ],
    "consumer_discretionary": [
        ("DFF", "Fed Funds Rate"),
        ("UMCSENT", "Consumer Sentiment"),
        ("PCE", "Personal Consumption Expenditures"),
        ("RSXFS", "Retail Sales ex Food Services"),
    ],
    "real_estate": [
        ("DFF", "Fed Funds Rate"),
        ("MORTGAGE30US", "30-Year Mortgage Rate"),
        ("HOUST", "Housing Starts"),
        ("CSUSHPISA", "Case-Shiller Home Price Index"),
    ],
    "default": [
        ("DFF", "Fed Funds Rate"),
        ("T10Y2Y", "10Y-2Y Yield Spread"),
        ("CPIAUCSL", "CPI All Items"),
        ("UNRATE", "Unemployment Rate"),
    ],
}


def fetch_macro_blob(sector: str, cache: Cache) -> dict:
    """Fetch latest values for sector-relevant FRED series."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return {"unavailable": True, "reason": "FRED_API_KEY not set"}

    key = f"fred_macro_{sector}"
    cached = cache.get(key, _FRED_TTL_H)
    if cached is not None:
        return cached

    series_list = SECTOR_SERIES.get(sector, SECTOR_SERIES["default"])
    result: dict[str, object] = {"sector": sector, "series": {}}

    for series_id, label in series_list:
        val = _latest_value(series_id, api_key)
        result["series"][series_id] = {"label": label, "latest_value": val}

    cache.set(key, result)
    return result


def _latest_value(series_id: str, api_key: str) -> Optional[float]:
    try:
        resp = requests.get(
            f"{_BASE}/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
            timeout=15,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if obs and obs[0]["value"] != ".":
            return float(obs[0]["value"])
    except Exception:
        pass
    return None

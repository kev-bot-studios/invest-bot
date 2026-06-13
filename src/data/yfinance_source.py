"""yfinance data source: prices, fundamentals, and news."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import yfinance as yf

from .cache import Cache


_PRICES_TTL_H = 24
_FUNDAMENTALS_TTL_H = 24 * 90  # ~1 quarter
_NEWS_TTL_H = 24


def fetch_prices(ticker: str, cache: Cache, period: str = "1y") -> dict:
    """Daily OHLCV history for momentum/volatility metrics."""
    key = f"yf_prices_{ticker}_{period}"
    cached = cache.get(key, _PRICES_TTL_H)
    if cached is not None:
        return cached

    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return {}

    result = {
        "dates": [str(d.date()) for d in hist.index],
        "close": hist["Close"].tolist(),
        "volume": hist["Volume"].tolist(),
    }
    cache.set(key, result)
    return result


def fetch_fundamentals(ticker: str, cache: Cache) -> dict:
    """Info dict + income/balance/cashflow statements."""
    key = f"yf_fundamentals_{ticker}"
    cached = cache.get(key, _FUNDAMENTALS_TTL_H)
    if cached is not None:
        return cached

    stock = yf.Ticker(ticker)
    info = stock.info or {}

    # Annual financials
    annual_income = _df_to_dict(stock.financials)
    annual_balance = _df_to_dict(stock.balance_sheet)
    annual_cashflow = _df_to_dict(stock.cashflow)

    # Quarterly financials
    q_income = _df_to_dict(stock.quarterly_financials)
    q_cashflow = _df_to_dict(stock.quarterly_cashflow)

    result = {
        "info": {k: _coerce(v) for k, v in info.items()},
        "annual_income": annual_income,
        "annual_balance": annual_balance,
        "annual_cashflow": annual_cashflow,
        "quarterly_income": q_income,
        "quarterly_cashflow": q_cashflow,
    }
    cache.set(key, result)
    return result


def fetch_news(ticker: str, cache: Cache) -> list[dict]:
    """Recent headlines (trailing ~30 days)."""
    key = f"yf_news_{ticker}"
    cached = cache.get(key, _NEWS_TTL_H)
    if cached is not None:
        return cached

    stock = yf.Ticker(ticker)
    raw_news = stock.news or []
    articles = []
    cutoff = datetime.now() - timedelta(days=30)

    for item in raw_news:
        pub = item.get("providerPublishTime")
        if pub:
            pub_dt = datetime.fromtimestamp(pub)
            if pub_dt < cutoff:
                continue
            pub_str = pub_dt.isoformat()
        else:
            pub_str = None

        content = item.get("content", {})
        articles.append({
            "title": content.get("title") or item.get("title", ""),
            "summary": content.get("summary") or "",
            "publisher": content.get("provider", {}).get("displayName") or item.get("publisher", ""),
            "published_at": pub_str,
            "url": content.get("canonicalUrl", {}).get("url") or item.get("link", ""),
        })

    cache.set(key, articles)
    return articles


def _df_to_dict(df) -> dict:
    if df is None or df.empty:
        return {}
    result = {}
    for col in df.columns:
        col_str = str(col.date()) if hasattr(col, "date") else str(col)
        result[col_str] = {}
        for idx in df.index:
            val = df.at[idx, col]
            try:
                import math
                if math.isnan(float(val)):
                    continue
                result[col_str][str(idx)] = float(val)
            except (TypeError, ValueError):
                result[col_str][str(idx)] = str(val) if val is not None else None
    return result


def _coerce(v: Any) -> Any:
    if isinstance(v, float):
        import math
        return None if math.isnan(v) else v
    return v

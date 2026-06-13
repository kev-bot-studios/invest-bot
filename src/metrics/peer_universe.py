"""Peer universe management — pull peer metrics and build percentile distributions."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from ..data.cache import Cache
from ..data import yfinance_source as yf_src
from ..data import edgar_source as edgar_src
from ..db import get_peer_universe, upsert_peer_universe


# Metric keys to build distributions for and the direction (higher=better or not)
_DISTRIBUTION_METRICS: list[tuple[str, str]] = [
    # valuation
    ("pe_ratio", "info.trailingPE"),
    ("ev_ebitda", "info.enterpriseToEbitda"),
    ("price_to_sales", "info.priceToSalesTrailing12Months"),
    ("price_to_book", "info.priceToBook"),
    # quality
    ("gross_margin", "derived.gross_margin"),
    ("operating_margin", "info.operatingMargins"),
    ("net_margin", "info.profitMargins"),
    ("roe", "info.returnOnEquity"),
    ("debt_to_equity", "info.debtToEquity"),
    ("fcf_margin", "derived.fcf_margin"),
    # growth
    ("revenue_yoy", "derived.revenue_yoy"),
    ("revenue_cagr_3y", "derived.revenue_cagr_3y"),
    # momentum
    ("return_3m", "derived.return_3m"),
    ("return_6m", "derived.return_6m"),
    ("return_12m", "derived.return_12m"),
    ("price_to_52w_high", "derived.price_to_52w_high"),
    # additional
    ("price_to_fcf", "derived.price_to_fcf"),
    ("ev_revenue", "derived.ev_revenue"),
    ("debt_to_ebitda", "derived.debt_to_ebitda"),
    ("roic", "derived.roic"),
    ("eps_cagr_3y", "derived.eps_cagr_3y"),
    ("fcf_growth_yoy", "derived.fcf_growth_yoy"),
]


def current_quarter() -> str:
    now = datetime.now()
    q = (now.month - 1) // 3 + 1
    return f"{now.year}-Q{q}"


def refresh_peer_universe(sector: str, peer_tickers: list[str],
                           cache: Cache, db_path: str) -> dict:
    """Fetch lightweight metrics for all peers and build distributions.

    Returns the distributions dict (also persisted to SQLite).
    """
    quarter = current_quarter()

    # Return cached version if available
    cached = get_peer_universe(db_path, sector, quarter)
    if cached:
        return cached

    peer_metrics: list[dict] = []

    for ticker in peer_tickers:
        try:
            m = _fetch_peer_metrics(ticker, cache)
            if m:
                peer_metrics.append(m)
        except Exception:
            continue

    distributions = _build_distributions(peer_metrics)

    upsert_peer_universe(db_path, sector, quarter, peer_tickers, distributions)
    return distributions


def _fetch_peer_metrics(ticker: str, cache: Cache) -> dict:
    """Light fetch — just yfinance info + EDGAR facts + 1y prices."""
    import math

    info_data = yf_src.fetch_fundamentals(ticker, cache).get("info", {})
    prices = yf_src.fetch_prices(ticker, cache)
    edgar = edgar_src.fetch_company_facts_timeseries(ticker, cache)

    def safe(v) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    closes = prices.get("close", [])

    def ret(n: int) -> Optional[float]:
        if len(closes) < n + 1:
            if len(closes) >= int(n * 0.95):
                c, p = closes[-1], closes[0]
                return (c - p) / p if p else None
            return None
        c = closes[-1]
        p = closes[-(n + 1)]
        return (c - p) / p if p else None

    rev_by_year = edgar.get("revenue_by_year", {})
    rev_vals = [v for _, v in sorted(rev_by_year.items(), reverse=True) if v is not None]

    market_cap = safe(info_data.get("marketCap"))
    fcf = safe(info_data.get("freeCashflow"))
    revenue = safe(info_data.get("totalRevenue"))
    gross_profit = safe(info_data.get("grossProfits"))
    ev = safe(info_data.get("enterpriseValue"))
    total_debt = safe(info_data.get("totalDebt"))
    ebitda = safe(info_data.get("ebitda"))
    total_assets = safe(info_data.get("totalAssets"))
    net_income = safe(info_data.get("netIncomeToCommon"))
    current_liab = safe(info_data.get("currentLiabilities")) or safe(info_data.get("totalCurrentLiabilities"))
    hi52 = safe(info_data.get("fiftyTwoWeekHigh"))
    current_price = safe(info_data.get("currentPrice")) or (closes[-1] if closes else None)

    def div(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    def cagr(final, initial, years):
        if final is None or initial is None or initial == 0 or years <= 0:
            return None
        if final / initial < 0:
            return None
        return (final / initial) ** (1 / years) - 1

    return {
        "ticker": ticker,
        "pe_ratio": safe(info_data.get("trailingPE")),
        "ev_ebitda": safe(info_data.get("enterpriseToEbitda")),
        "price_to_sales": safe(info_data.get("priceToSalesTrailing12Months")),
        "price_to_book": safe(info_data.get("priceToBook")),
        "price_to_fcf": div(market_cap, fcf),
        "ev_revenue": div(ev, revenue),
        "gross_margin": div(gross_profit, revenue),
        "operating_margin": safe(info_data.get("operatingMargins")),
        "net_margin": safe(info_data.get("profitMargins")),
        "roe": safe(info_data.get("returnOnEquity")),
        "roic": div(net_income, (total_assets - current_liab) if (total_assets and current_liab) else None),
        "debt_to_equity": safe(info_data.get("debtToEquity")),
        "debt_to_ebitda": div(total_debt, ebitda),
        "fcf_margin": div(fcf, revenue),
        "revenue_yoy": (rev_vals[0] / rev_vals[1] - 1) if len(rev_vals) > 1 and rev_vals[1] else None,
        "revenue_cagr_3y": cagr(rev_vals[0] if rev_vals else None, rev_vals[3] if len(rev_vals) > 3 else None, 3),
        "eps_cagr_3y": None,  # Skipped for speed in peer pull
        "fcf_growth_yoy": None,
        "return_3m": ret(63),
        "return_6m": ret(126),
        "return_12m": ret(252),
        "price_to_52w_high": div(current_price, hi52),
    }


def _build_distributions(peer_metrics: list[dict]) -> dict:
    """Build a distribution list for each metric key."""
    distributions: dict[str, list[float]] = {}
    if not peer_metrics:
        return distributions

    keys = [k for k in peer_metrics[0] if k != "ticker"]
    for k in keys:
        vals = [m[k] for m in peer_metrics if m.get(k) is not None]
        distributions[k] = [float(v) for v in vals]

    return distributions

"""Pure-Python metrics engine. No LLM. No rounding surprises.

All ratios, growth rates, and factor scores are computed here.
Scores are percentile-ranked against the peer universe distributions.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Optional[float]:
    """Return float or None; swallow NaN/inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


def _cagr(final: Optional[float], initial: Optional[float], years: float) -> Optional[float]:
    if final is None or initial is None or initial == 0 or years <= 0:
        return None
    if final / initial < 0:
        return None  # sign flip — not meaningful as CAGR
    return (final / initial) ** (1 / years) - 1


def _percentile_rank(value: float, distribution: list[float]) -> float:
    """Return the percentile (0–100) of value within the distribution."""
    if not distribution:
        return 50.0
    arr = np.array(distribution, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 50.0
    return float(np.mean(arr <= value) * 100)


def _quintile_score(value: Optional[float], distribution: list[float],
                    higher_is_better: bool = True) -> Optional[int]:
    """Map a value to a 1–5 quintile score against the peer distribution."""
    if value is None or not distribution:
        return None
    pct = _percentile_rank(value, distribution)
    if higher_is_better:
        if pct >= 80:
            return 5
        elif pct >= 60:
            return 4
        elif pct >= 40:
            return 3
        elif pct >= 20:
            return 2
        else:
            return 1
    else:  # lower is better (e.g., P/E, debt ratio)
        if pct <= 20:
            return 5
        elif pct <= 40:
            return 4
        elif pct <= 60:
            return 3
        elif pct <= 80:
            return 2
        else:
            return 1


def _avg_score(scores: list[Optional[int]]) -> Optional[float]:
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


# ---------------------------------------------------------------------------
# Metrics computation from raw snapshot data
# ---------------------------------------------------------------------------

def compute_metrics(snapshot: dict) -> dict:
    """Derive all computable metrics from the raw data snapshot."""
    info = snapshot.get("yf_info", {})
    prices = snapshot.get("prices", {})
    annual_income = snapshot.get("annual_income", {})
    annual_cashflow = snapshot.get("annual_cashflow", {})
    edgar_ts = snapshot.get("edgar_timeseries", {})
    insider_txs = snapshot.get("insider_transactions", [])

    metrics: dict[str, Any] = {}

    # --- Valuation ---
    metrics["pe_ratio"] = _safe(info.get("trailingPE"))
    metrics["forward_pe"] = _safe(info.get("forwardPE"))
    metrics["ev_ebitda"] = _safe(info.get("enterpriseToEbitda"))
    metrics["price_to_sales"] = _safe(info.get("priceToSalesTrailing12Months"))
    metrics["price_to_book"] = _safe(info.get("priceToBook"))

    # P/FCF
    market_cap = _safe(info.get("marketCap"))
    metrics["market_cap"] = market_cap
    fcf = _safe(info.get("freeCashflow"))
    metrics["price_to_fcf"] = (market_cap / fcf) if market_cap and fcf and fcf > 0 else None

    # EV / Revenue (fallback when earnings negative)
    ev = _safe(info.get("enterpriseValue"))
    revenue = _safe(info.get("totalRevenue"))
    metrics["ev_revenue"] = (ev / revenue) if ev and revenue and revenue > 0 else None

    metrics["has_negative_earnings"] = (
        (metrics["pe_ratio"] is None or metrics["pe_ratio"] < 0) and
        _safe(info.get("netIncomeToCommon")) is not None and
        (_safe(info.get("netIncomeToCommon")) or 0) < 0
    )

    # --- Quality ---
    gross_profit = _safe(info.get("grossProfits"))
    net_income = _safe(info.get("netIncomeToCommon"))
    operating_cf = _safe(info.get("operatingCashflow"))
    total_debt = _safe(info.get("totalDebt"))
    equity = _safe(info.get("bookValue")) and _safe(info.get("sharesOutstanding")) and (
        (_safe(info.get("bookValue")) or 0) * (_safe(info.get("sharesOutstanding")) or 0)
    )
    ebitda = _safe(info.get("ebitda"))

    metrics["gross_margin"] = (gross_profit / revenue) if gross_profit and revenue and revenue > 0 else None
    metrics["operating_margin"] = _safe(info.get("operatingMargins"))
    metrics["net_margin"] = _safe(info.get("profitMargins"))
    metrics["roe"] = _safe(info.get("returnOnEquity"))
    metrics["roic"] = None  # Computed below if possible
    metrics["debt_to_equity"] = _safe(info.get("debtToEquity"))
    metrics["debt_to_ebitda"] = (total_debt / ebitda) if total_debt and ebitda and ebitda > 0 else None
    metrics["current_ratio"] = _safe(info.get("currentRatio"))
    metrics["fcf_margin"] = (fcf / revenue) if fcf and revenue and revenue > 0 else None

    # ROIC approximation: EBIT*(1-t) / (Total Assets - Current Liabilities)
    # We'll use net_income / (total_assets - current_liabilities) as a proxy
    total_assets = _safe(info.get("totalAssets"))
    current_liab = _safe(info.get("currentLiabilities")) or _safe(info.get("totalCurrentLiabilities"))
    if net_income and total_assets and current_liab and (total_assets - current_liab) > 0:
        metrics["roic"] = net_income / (total_assets - current_liab)

    # --- Growth (from EDGAR time series, fallback to yfinance annuals) ---
    rev_by_year = edgar_ts.get("revenue_by_year", {})
    ni_by_year = edgar_ts.get("net_income_by_year", {})

    # Flatten to sorted descending lists
    rev_vals = [v for _, v in sorted(rev_by_year.items(), reverse=True) if v is not None]
    ni_vals = [v for _, v in sorted(ni_by_year.items(), reverse=True) if v is not None]

    # Fallback to yfinance quarterly income
    if not rev_vals:
        ann = annual_income or {}
        rev_vals = [d.get("Total Revenue") for _, d in sorted(ann.items(), reverse=True) if isinstance(d, dict)]
        rev_vals = [v for v in rev_vals if v is not None]

    # Data-depth: how many annual fiscal periods back the growth factor
    metrics["annual_periods"] = len(rev_vals)

    # YoY growth (latest vs prior year)
    metrics["revenue_yoy"] = _pct_change(rev_vals[0] if len(rev_vals) > 0 else None,
                                          rev_vals[1] if len(rev_vals) > 1 else None)
    metrics["net_income_yoy"] = _pct_change(ni_vals[0] if len(ni_vals) > 0 else None,
                                              ni_vals[1] if len(ni_vals) > 1 else None)

    # 3-year CAGR
    metrics["revenue_cagr_3y"] = _cagr(rev_vals[0] if len(rev_vals) > 0 else None,
                                         rev_vals[3] if len(rev_vals) > 3 else None, 3.0)
    metrics["eps_cagr_3y"] = _cagr(ni_vals[0] if len(ni_vals) > 0 else None,
                                     ni_vals[3] if len(ni_vals) > 3 else None, 3.0)

    # EPS YoY from yfinance info
    trailing_eps = _safe(info.get("trailingEps"))
    forward_eps = _safe(info.get("forwardEps"))
    metrics["eps_forward_growth"] = _pct_change(forward_eps, trailing_eps)

    # FCF growth (from cashflow statement if available)
    metrics["fcf_growth_yoy"] = None
    if annual_cashflow:
        dates = sorted(annual_cashflow.keys(), reverse=True)
        if len(dates) >= 2:
            fcf_cur = annual_cashflow.get(dates[0], {}).get("Free Cash Flow")
            fcf_prv = annual_cashflow.get(dates[1], {}).get("Free Cash Flow")
            metrics["fcf_growth_yoy"] = _pct_change(_safe(fcf_cur), _safe(fcf_prv))

    # --- Momentum (from price history) ---
    closes = prices.get("close", [])
    dates_list = prices.get("dates", [])
    metrics["price_history_days"] = len(closes)

    def _ret_n_days(n: int) -> Optional[float]:
        # A year of trading days is ~251; tolerate slightly short histories
        if len(closes) < n + 1:
            if len(closes) >= int(n * 0.95):
                return _pct_change(closes[-1], closes[0])
            return None
        return _pct_change(closes[-1], closes[-(n + 1)])

    metrics["return_3m"] = _ret_n_days(63)
    metrics["return_6m"] = _ret_n_days(126)
    metrics["return_12m"] = _ret_n_days(252)

    hi52 = _safe(info.get("fiftyTwoWeekHigh"))
    current_price = _safe(info.get("currentPrice")) or (closes[-1] if closes else None)
    metrics["price_to_52w_high"] = (current_price / hi52) if current_price and hi52 and hi52 > 0 else None
    metrics["current_price"] = current_price

    # Volatility (annualised stddev of daily returns)
    metrics["volatility_annual"] = None
    if len(closes) >= 30:
        arr = np.array(closes[-252:], dtype=float)
        daily_rets = np.diff(arr) / arr[:-1]
        metrics["volatility_annual"] = float(np.std(daily_rets) * np.sqrt(252))

    # --- Insider activity (from Form 4, trailing 6m) ---
    buy_value = sum(t["value"] for t in insider_txs if t.get("is_buy"))
    sell_value = sum(t["value"] for t in insider_txs if not t.get("is_buy"))
    buy_count = sum(1 for t in insider_txs if t.get("is_buy"))
    sell_count = sum(1 for t in insider_txs if not t.get("is_buy"))
    net_insider_value = buy_value - sell_value

    # Cluster detection: >2 distinct buyers in trailing 30 days
    recent_buys = [t for t in insider_txs if t.get("is_buy")]
    metrics["insider_buy_value"] = buy_value
    metrics["insider_sell_value"] = sell_value
    metrics["insider_net_value"] = net_insider_value
    metrics["insider_buy_count"] = buy_count
    metrics["insider_sell_count"] = sell_count
    metrics["insider_cluster"] = buy_count >= 3

    return metrics


# ---------------------------------------------------------------------------
# Trend computation (multi-year and quarterly time series)
# ---------------------------------------------------------------------------

def compute_trends(snapshot: dict) -> dict:
    """Build annual, quarterly, and price trend series from the snapshot.

    Returns:
        {
          "annual": [{fiscal_year_end, revenue, net_income, net_margin,
                      revenue_yoy, net_income_yoy}, ...]  # newest first
          "quarterly": [{quarter_end, revenue, net_income, net_margin}, ...]
          "price": {ma_50, ma_200, price_vs_ma50, price_vs_ma200,
                    returns: {1m, 3m, 6m, 12m}}
        }
    """
    edgar_ts = snapshot.get("edgar_timeseries", {})
    q_income = snapshot.get("quarterly_income", {})
    annual_income = snapshot.get("annual_income", {})
    prices = snapshot.get("prices", {})

    # --- Annual series (EDGAR is authoritative; yfinance as fallback) ---
    rev_by_year = dict(edgar_ts.get("revenue_by_year", {}))
    ni_by_year = dict(edgar_ts.get("net_income_by_year", {}))

    if not rev_by_year and annual_income:
        for date_str, items in annual_income.items():
            if isinstance(items, dict):
                rev = _safe(items.get("Total Revenue"))
                ni = _safe(items.get("Net Income"))
                if rev is not None:
                    rev_by_year[date_str] = rev
                if ni is not None:
                    ni_by_year[date_str] = ni

    years = sorted(rev_by_year.keys(), reverse=True)[:6]
    annual: list[dict] = []
    for i, y in enumerate(years):
        rev = _safe(rev_by_year.get(y))
        ni = _safe(ni_by_year.get(y))
        prev_y = years[i + 1] if i + 1 < len(years) else None
        annual.append({
            "fiscal_year_end": y,
            "revenue": rev,
            "net_income": ni,
            "net_margin": (ni / rev) if (ni is not None and rev) else None,
            "revenue_yoy": _pct_change(rev, _safe(rev_by_year.get(prev_y))) if prev_y else None,
            "net_income_yoy": _pct_change(ni, _safe(ni_by_year.get(prev_y))) if prev_y else None,
        })

    # --- Quarterly series (yfinance quarterly income, newest first) ---
    quarterly: list[dict] = []
    for date_str in sorted(q_income.keys(), reverse=True)[:8]:
        items = q_income.get(date_str, {})
        if not isinstance(items, dict):
            continue
        rev = _safe(items.get("Total Revenue"))
        ni = _safe(items.get("Net Income"))
        quarterly.append({
            "quarter_end": date_str,
            "revenue": rev,
            "net_income": ni,
            "net_margin": (ni / rev) if (ni is not None and rev) else None,
        })

    # --- Price trend ---
    closes = prices.get("close", [])
    price_trend: dict[str, Any] = {}
    if closes:
        current = closes[-1]
        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else None
        ma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else None
        price_trend = {
            "ma_50": ma50,
            "ma_200": ma200,
            "price_vs_ma50": _pct_change(current, ma50) if ma50 else None,
            "price_vs_ma200": _pct_change(current, ma200) if ma200 else None,
            "returns": {
                "1m": _pct_change(current, closes[-22]) if len(closes) > 22 else None,
                "3m": _pct_change(current, closes[-64]) if len(closes) > 64 else None,
                "6m": _pct_change(current, closes[-127]) if len(closes) > 127 else None,
                "12m": (_pct_change(current, closes[-252]) if len(closes) >= 252
                        else _pct_change(current, closes[0]) if len(closes) >= 240
                        else None),
            },
        }

    return {"annual": annual, "quarterly": quarterly, "price": price_trend}


# ---------------------------------------------------------------------------
# Market positioning context (short interest, analyst consensus)
# ---------------------------------------------------------------------------

def compute_market_context(snapshot: dict) -> dict:
    """Unscored positioning context from the yfinance info dict.

    These are interpretation inputs for synthesis, not factor scores:
    short interest is ambiguous (bearish bet vs squeeze fuel) and analyst
    targets are consensus context, not a deterministic signal.
    """
    info = snapshot.get("yf_info", {})
    current = _safe(info.get("currentPrice"))

    short_now = _safe(info.get("sharesShort"))
    short_prior = _safe(info.get("sharesShortPriorMonth"))
    target_mean = _safe(info.get("targetMeanPrice"))

    return {
        "short_interest": {
            "pct_of_float": _safe(info.get("shortPercentOfFloat")),
            "days_to_cover": _safe(info.get("shortRatio")),
            "shares_short": short_now,
            "shares_short_prior_month": short_prior,
            "mom_change": _pct_change(short_now, short_prior),
        },
        "analyst_consensus": {
            "target_mean": target_mean,
            "target_high": _safe(info.get("targetHighPrice")),
            "target_low": _safe(info.get("targetLowPrice")),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "recommendation_mean": _safe(info.get("recommendationMean")),
            "upside_to_target": _pct_change(target_mean, current)
            if (target_mean and current) else None,
        },
    }


# ---------------------------------------------------------------------------
# Percentile scoring against peer distributions
# ---------------------------------------------------------------------------

def score_metrics(metrics: dict, distributions: dict) -> dict:
    """Map each metric to a 1–5 score using peer quintile distributions."""
    scores: dict[str, Any] = {}

    def _score(metric_key: str, dist_key: str, higher_is_better: bool) -> Optional[int]:
        val = metrics.get(metric_key)
        dist = distributions.get(dist_key, [])
        return _quintile_score(val, dist, higher_is_better)

    # Valuation — lower ratio = better value
    neg_earnings = metrics.get("has_negative_earnings", False)
    if neg_earnings:
        val_scores = [
            _score("price_to_sales", "price_to_sales", False),
            _score("ev_revenue", "ev_revenue", False),
            _score("price_to_fcf", "price_to_fcf", False),
        ]
        scores["valuation_note"] = "P/E unavailable (negative earnings); using P/S, EV/Rev, P/FCF"
    else:
        val_scores = [
            _score("pe_ratio", "pe_ratio", False),
            _score("ev_ebitda", "ev_ebitda", False),
            _score("price_to_sales", "price_to_sales", False),
            _score("price_to_fcf", "price_to_fcf", False),
        ]
    scores["valuation"] = _avg_score(val_scores)

    # Quality — higher margins/returns = better; lower debt = better
    quality_scores = [
        _score("gross_margin", "gross_margin", True),
        _score("operating_margin", "operating_margin", True),
        _score("net_margin", "net_margin", True),
        _score("roe", "roe", True),
        _score("roic", "roic", True),
        _score("debt_to_ebitda", "debt_to_ebitda", False),
        _score("fcf_margin", "fcf_margin", True),
    ]
    scores["quality"] = _avg_score(quality_scores)

    # Growth — higher = better
    growth_scores = [
        _score("revenue_yoy", "revenue_yoy", True),
        _score("revenue_cagr_3y", "revenue_cagr_3y", True),
        _score("eps_cagr_3y", "eps_cagr_3y", True),
        _score("fcf_growth_yoy", "fcf_growth_yoy", True),
    ]
    scores["growth"] = _avg_score(growth_scores)

    # Momentum — higher returns = better; low volatility is neutral (not scored separately)
    mom_scores = [
        _score("return_3m", "return_3m", True),
        _score("return_6m", "return_6m", True),
        _score("return_12m", "return_12m", True),
        _score("price_to_52w_high", "price_to_52w_high", True),
    ]
    scores["momentum"] = _avg_score(mom_scores)

    # Insider — net buying is a positive signal. Selling is normalized by
    # market cap: routine comp-plan selling at mega caps shouldn't tank the score.
    net_val = metrics.get("insider_net_value", 0) or 0
    buy_ct = metrics.get("insider_buy_count", 0) or 0
    sell_ct = metrics.get("insider_sell_count", 0) or 0
    market_cap = metrics.get("market_cap")
    if buy_ct + sell_ct == 0:
        scores["insider"] = 3  # no activity = neutral
    elif net_val > 0:
        cluster = metrics.get("insider_cluster", False)
        scores["insider"] = 5 if cluster else 4
    else:
        # Net selling: scale severity by % of market cap (fallback: absolute value)
        if market_cap and market_cap > 0:
            sell_frac = abs(net_val) / market_cap
            if sell_frac < 0.0001:    # <0.01% of mcap — routine
                scores["insider"] = 3
            elif sell_frac < 0.001:   # <0.1% — moderate
                scores["insider"] = 2
            else:                     # heavy selling
                scores["insider"] = 1
        else:
            scores["insider"] = 2 if net_val > -10_000_000 else 1

    return scores


def metric_percentiles(metrics: dict, distributions: dict) -> dict[str, float]:
    """Peer percentile (0–100) for each metric that has a distribution."""
    out: dict[str, float] = {}
    for key, dist in distributions.items():
        val = metrics.get(key)
        if val is not None and dist:
            out[key] = round(_percentile_rank(float(val), dist), 0)
    return out


def compute_composite(scores: dict, weights_cfg) -> Optional[float]:
    """Weighted average of all factor scores that are present."""
    weight_map = {
        "valuation": weights_cfg.valuation,
        "quality": weights_cfg.quality,
        "growth": weights_cfg.growth,
        "momentum": weights_cfg.momentum,
        "insider": weights_cfg.insider,
        "filing_tone": weights_cfg.filing_tone,
        "news_tone": weights_cfg.news_tone,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for factor, weight in weight_map.items():
        val = scores.get(factor)
        if val is not None:
            weighted_sum += val * weight
            total_weight += weight
    if total_weight == 0:
        return None
    return round(weighted_sum / total_weight, 2)


def data_depth(metrics: dict) -> dict:
    """Classify how much history backs the history-dependent factors.

    Two independent axes:
      * Growth depth — number of annual fiscal periods on file (YoY needs 2, a
        3y CAGR needs 4). This is the real "how long public" proxy, since a
        recent IPO has only 1–2 annual filings.
      * Momentum availability — whether we have ~1y of prices for the 12-month
        return. NOTE: the price fetch is capped at ~1y by design, so price
        history measures *computability of momentum*, not company age — a name
        public for decades still returns ~251 days here. We reuse the same 0.95
        tolerance as the 12-month return so a normal 251-day year counts as full.

    The weaker axis drives the label. Composites already renormalize over the
    factors that have data, so this is purely a confidence signal: it tells you
    when a low growth/momentum score is missing-data rather than genuinely weak.
    """
    price_days = metrics.get("price_history_days", 0) or 0
    fiscal_years = metrics.get("annual_periods", 0) or 0
    momentum_ok = price_days >= int(252 * 0.95)   # ~239d; matches return_12m tolerance
    if fiscal_years < 2 or not momentum_ok:
        label = "thin"   # <2 fiscal years, or <~1y of prices → growth/momentum unreliable
    elif fiscal_years < 4:
        label = "ltd"    # has YoY + momentum, but not a full 3y trend yet
    else:
        label = "full"
    return {
        "price_days": price_days,
        "fiscal_years": fiscal_years,
        "momentum_ok": momentum_ok,
        "label": label,
    }

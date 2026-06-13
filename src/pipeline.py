"""Pipeline orchestration — ties all layers together for a single run."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import RunConfig, load_peer_tickers
from .data.cache import Cache
from .data import yfinance_source as yf_src
from .data import edgar_source as edgar_src
from .data import fred_source as fred_src
from .data import finnhub_source as finnhub_src
from .db import (
    init_db, insert_run, update_run_status,
    upsert_ticker, upsert_output,
)
from .llm.client import LLMClient
from .llm import extraction, synthesis as synth_mod
from .metrics import engine as metrics_eng
from .metrics.peer_universe import refresh_peer_universe


@dataclass
class TickerResult:
    ticker: str
    snapshot: dict
    metrics: dict
    scores: dict
    extractions: dict
    synthesis: Optional[dict]
    report_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    ticker_results: list[TickerResult]
    comparative: Optional[dict]
    peer_distributions: dict


def run(cfg: RunConfig, output_dir: str = "reports") -> RunResult:
    """Execute the full pipeline for all configured tickers."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    started_dt = datetime.now()
    started_at = started_dt.isoformat()
    run_stamp = started_dt.strftime("%Y-%m-%d_%H%M%S")  # filesystem-safe report suffix
    cache = Cache(cfg.cache_dir)

    init_db(cfg.db_path)
    insert_run(cfg.db_path, run_id, started_at, {
        "tickers": cfg.tickers, "sector": cfg.sector,
        "prompt_version": cfg.prompt_version,
    }, cfg.prompt_version)

    llm_client = LLMClient(cfg.llm_providers)
    provider = cfg.first_available_provider()
    llm_available = provider is not None

    # ------------------------------------------------------------------
    # 1. Peer universe refresh
    # ------------------------------------------------------------------
    print(f"\n[run:{run_id}] Refreshing peer universe for sector '{cfg.sector}'...")
    peer_tickers = load_peer_tickers(cfg.sector)
    if not peer_tickers:
        peer_tickers = cfg.tickers  # fallback: score against each other

    peer_distributions: dict = {}
    if cfg.sources.edgar_companyfacts or cfg.sources.yfinance_fundamentals:
        try:
            peer_distributions = refresh_peer_universe(
                cfg.sector, peer_tickers, cache, cfg.db_path
            )
            print(f"  Peer universe: {len(peer_tickers)} tickers, "
                  f"{len(peer_distributions)} metric distributions.")
        except Exception as e:
            print(f"  Warning: peer universe refresh failed: {e}")

    # ------------------------------------------------------------------
    # 2. Macro blob (shared across all tickers)
    # ------------------------------------------------------------------
    macro_blob: dict = {}
    if cfg.sources.fred:
        print(f"[run:{run_id}] Fetching FRED macro data...")
        macro_blob = fred_src.fetch_macro_blob(cfg.sector, cache)
        if macro_blob.get("unavailable"):
            print(f"  FRED unavailable: {macro_blob.get('reason')}")

    # ------------------------------------------------------------------
    # 3. Per-ticker pipeline
    # ------------------------------------------------------------------
    ticker_results: list[TickerResult] = []

    for ticker in cfg.tickers:
        print(f"\n[run:{run_id}] Processing {ticker}...")
        result = _process_ticker(
            ticker=ticker,
            cfg=cfg,
            cache=cache,
            peer_distributions=peer_distributions,
            macro_blob=macro_blob,
            llm_client=llm_client,
            llm_available=llm_available,
            run_id=run_id,
            run_stamp=run_stamp,
            output_dir=output_dir,
        )
        ticker_results.append(result)

    # ------------------------------------------------------------------
    # 4. Comparative call (multi-ticker)
    # ------------------------------------------------------------------
    comparative = None
    if llm_available and len(ticker_results) > 1:
        print(f"\n[run:{run_id}] Running comparative analysis...")
        comp_inputs = [
            {
                "ticker": r.ticker,
                "metrics": r.metrics,
                "scores": r.scores,
                "extractions": r.extractions,
            }
            for r in ticker_results
        ]
        try:
            comparative = synth_mod.compare_tickers(
                comp_inputs, llm_client, run_id, cfg.db_path
            )
            if comparative:
                print(f"  Comparative ranking: {comparative.get('overall_ranking')}")
                comp_md = _render_comparative_md(comparative)
                for r in ticker_results:
                    upsert_output(cfg.db_path, run_id, r.ticker,
                                  None, comp_md)
        except Exception as e:
            print(f"  Comparative call failed: {e}")

    update_run_status(cfg.db_path, run_id, "completed",
                      provider.name if provider else None)

    return RunResult(
        run_id=run_id,
        ticker_results=ticker_results,
        comparative=comparative,
        peer_distributions=peer_distributions,
    )


def _process_ticker(
    ticker: str,
    cfg: RunConfig,
    cache: Cache,
    peer_distributions: dict,
    macro_blob: dict,
    llm_client: LLMClient,
    llm_available: bool,
    run_id: str,
    run_stamp: str,
    output_dir: str,
) -> TickerResult:
    errors: list[str] = []
    snapshot: dict = {}

    # When this ticker's pull began (UTC). Note: with the TTL cache, individual
    # sources may return data fetched in an earlier run; this stamps the run.
    fetched_at = datetime.now(timezone.utc).isoformat()

    # --- yfinance data ---
    info: dict = {}
    prices: dict = {}
    news: list = []
    company_name = ticker
    description = ""

    if cfg.sources.yfinance_fundamentals:
        try:
            fundamentals = yf_src.fetch_fundamentals(ticker, cache)
            info = fundamentals.get("info", {})
            snapshot["yf_info"] = info
            snapshot["annual_income"] = fundamentals.get("annual_income", {})
            snapshot["annual_balance"] = fundamentals.get("annual_balance", {})
            snapshot["annual_cashflow"] = fundamentals.get("annual_cashflow", {})
            snapshot["quarterly_income"] = fundamentals.get("quarterly_income", {})
            company_name = info.get("longName", ticker)
            description = info.get("longBusinessSummary", "")
            print(f"  yfinance fundamentals: OK ({company_name})")
        except Exception as e:
            errors.append(f"yfinance_fundamentals: {e}")
            print(f"  yfinance fundamentals: FAILED ({e})")

    if cfg.sources.yfinance_prices:
        try:
            prices = yf_src.fetch_prices(ticker, cache)
            snapshot["prices"] = prices
            print(f"  yfinance prices: OK ({len(prices.get('close', []))} days)")
        except Exception as e:
            errors.append(f"yfinance_prices: {e}")
            print(f"  yfinance prices: FAILED ({e})")

    if cfg.sources.yfinance_news:
        try:
            news = yf_src.fetch_news(ticker, cache)
            snapshot["news"] = news
            print(f"  yfinance news: OK ({len(news)} articles)")
        except Exception as e:
            errors.append(f"yfinance_news: {e}")
            print(f"  yfinance news: FAILED ({e})")

    # --- EDGAR data ---
    edgar_ts: dict = {}
    current_sections: dict = {}
    prior_sections: dict = {}
    insider_txs: list = []

    if cfg.sources.edgar_companyfacts:
        try:
            edgar_ts = edgar_src.fetch_company_facts_timeseries(ticker, cache)
            snapshot["edgar_timeseries"] = edgar_ts
            print(f"  EDGAR companyfacts: OK")
        except Exception as e:
            errors.append(f"edgar_companyfacts: {e}")
            print(f"  EDGAR companyfacts: FAILED ({e})")

    if cfg.sources.edgar_filings:
        try:
            current_sections = edgar_src.fetch_latest_filing_sections(
                ticker, "10-K", cache, which=0
            )
            snapshot["current_10k_sections"] = current_sections
            # Prior-year 10-K enables true year-over-year tone-delta analysis
            prior_sections = edgar_src.fetch_latest_filing_sections(
                ticker, "10-K", cache, which=1
            )
            snapshot["prior_10k_sections"] = prior_sections
            has_mda = bool(current_sections.get("item_7"))
            has_risks = bool(current_sections.get("item_1a"))
            has_prior = bool(prior_sections.get("item_7") or prior_sections.get("item_1a"))
            print(f"  EDGAR filings (10-K): OK (MD&A={'yes' if has_mda else 'no'}, "
                  f"Risks={'yes' if has_risks else 'no'}, "
                  f"prior-year={'yes' if has_prior else 'no'})")
        except Exception as e:
            errors.append(f"edgar_filings: {e}")
            print(f"  EDGAR filings: FAILED ({e})")

    if cfg.sources.edgar_form4:
        try:
            insider_txs = edgar_src.fetch_insider_transactions(ticker, cache)
            snapshot["insider_transactions"] = insider_txs
            buy_ct = sum(1 for t in insider_txs if t.get("is_buy"))
            sell_ct = sum(1 for t in insider_txs if not t.get("is_buy"))
            print(f"  EDGAR Form 4: OK ({buy_ct} buys, {sell_ct} sells)")
        except Exception as e:
            errors.append(f"edgar_form4: {e}")
            print(f"  EDGAR Form 4: FAILED ({e})")

    # --- Finnhub ---
    analyst_blob: dict = {}
    if cfg.sources.finnhub:
        try:
            analyst_blob = finnhub_src.fetch_analyst_blob(ticker, cache)
            if analyst_blob.get("unavailable"):
                print(f"  Finnhub: unavailable ({analyst_blob.get('reason')})")
            else:
                print(f"  Finnhub: OK")
        except Exception as e:
            errors.append(f"finnhub: {e}")

    # --- Metrics engine ---
    metrics = metrics_eng.compute_metrics(snapshot)
    trends = metrics_eng.compute_trends(snapshot)
    metrics["trends"] = trends  # persisted with metrics; also reaches synthesis
    metrics["market_context"] = metrics_eng.compute_market_context(snapshot)
    percentiles = metrics_eng.metric_percentiles(metrics, peer_distributions)
    metrics["peer_percentiles"] = percentiles
    scores = metrics_eng.score_metrics(metrics, peer_distributions)
    scores["composite"] = metrics_eng.compute_composite(scores, cfg.weights)
    print(f"  Metrics computed. Composite score: {scores.get('composite')}")

    # Persist snapshot + metrics + scores
    upsert_ticker(
        cfg.db_path, run_id, ticker,
        info.get("sector", cfg.sector),
        snapshot, metrics, scores,
        metrics.get("current_price"),
        fetched_at,
    )

    # --- LLM extraction ---
    extractions: dict = {}

    if llm_available:
        # Reset so we report the model actually used for THIS ticker's calls.
        llm_client.last_provider = None
        llm_client.last_model = None
        if cfg.sources.yfinance_news and news:
            print(f"  LLM: news tone extraction...")
            tone = extraction.extract_news_tone(
                ticker, company_name, news, llm_client, run_id, cfg.db_path
            )
            if tone:
                extractions["news_tone"] = tone
                scores["news_tone"] = tone.get("score")
            else:
                extractions["news_tone"] = {"unavailable": True}

        if cfg.sources.edgar_filings and (current_sections or prior_sections):
            print(f"  LLM: MD&A delta extraction...")
            mda = extraction.extract_mda_delta(
                ticker, company_name, current_sections, prior_sections,
                llm_client, run_id, cfg.db_path
            )
            if mda:
                extractions["mda_delta"] = mda

            print(f"  LLM: risk diff extraction...")
            rdiff = extraction.extract_risk_diff(
                ticker, company_name, current_sections, prior_sections,
                llm_client, run_id, cfg.db_path
            )
            if rdiff:
                extractions["risk_diff"] = rdiff

            # Composite filing tone score = avg of mda + risk
            mda_score = mda.get("score") if mda else None
            rdiff_score = rdiff.get("score") if rdiff else None
            valid = [s for s in [mda_score, rdiff_score] if s is not None]
            if valid:
                scores["filing_tone"] = round(sum(valid) / len(valid), 1)
                scores["composite"] = metrics_eng.compute_composite(scores, cfg.weights)

        # --- Synthesis ---
        # Finnhub is optional; fall back to the free yfinance analyst consensus
        # so the prompt's analyst slot is populated without a Finnhub key.
        synth_analyst = analyst_blob
        if not analyst_blob or analyst_blob.get("unavailable"):
            synth_analyst = metrics.get("market_context", {}).get("analyst_consensus", {})

        print(f"  LLM: synthesis...")
        synthesis_result = synth_mod.synthesize_ticker(
            ticker=ticker,
            company_name=company_name,
            description=description,
            metrics=metrics,
            scores=scores,
            extractions=extractions,
            macro_blob=macro_blob,
            analyst_blob=synth_analyst,
            client=llm_client,
            run_id=run_id,
            db_path=cfg.db_path,
        )
    else:
        synthesis_result = None
        if not llm_available:
            print(f"  LLM: skipped (no provider API key configured)")

    # Model actually used for this ticker's LLM calls (None if all failed/skipped).
    llm_model_used = None
    if llm_client.last_model:
        llm_model_used = f"{llm_client.last_provider} / {llm_client.last_model}"

    # --- Render and save report ---
    report_md = _render_report_md(
        ticker, company_name, description, metrics, scores,
        extractions, synthesis_result, errors, llm_model_used,
    )
    report_path = f"{output_dir}/{ticker}_{run_stamp}_{run_id}.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    upsert_output(cfg.db_path, run_id, ticker, report_md)
    print(f"  Report saved: {report_path}")

    return TickerResult(
        ticker=ticker,
        snapshot=snapshot,
        metrics=metrics,
        scores=scores,
        extractions=extractions,
        synthesis=synthesis_result,
        report_path=report_path,
        errors=errors,
    )


def _render_report_md(
    ticker: str,
    company_name: str,
    description: str,
    metrics: dict,
    scores: dict,
    extractions: dict,
    synthesis: Optional[dict],
    errors: list[str],
    llm_model: Optional[str] = None,
) -> str:
    from datetime import date

    lines = [
        f"# {ticker} — Equity Analysis Report",
        f"**{company_name}** | {date.today().isoformat()}",
        "",
        "> **Disclaimer:** This is a research aid, not investment advice.",
        "",
    ]
    if llm_model:
        lines += [f"_LLM-judged scores and analysis generated by: **{llm_model}**_", ""]
    lines += [
        "---",
        "",
        "## Factor Scorecard",
        "",
        "| Factor | Score (1–5) | Type |",
        "|--------|-------------|------|",
    ]

    factor_labels = [
        ("valuation", "Valuation", "Computed"),
        ("quality", "Quality", "Computed"),
        ("growth", "Growth", "Computed"),
        ("momentum", "Momentum", "Computed"),
        ("insider", "Insider Activity", "Computed"),
        ("filing_tone", "Filing Tone", "LLM-judged"),
        ("news_tone", "News Tone", "LLM-judged"),
    ]
    for key, label, kind in factor_labels:
        val = scores.get(key)
        score_str = f"{val:.1f}" if val is not None else "N/A"
        lines.append(f"| {label} | {score_str} | {kind} |")

    composite = scores.get("composite")
    lines.append("")
    lines.append(f"**Composite score:** {composite:.2f} / 5.00" if composite else "**Composite score:** N/A")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Key metrics table with peer percentile context
    pctls = metrics.get("peer_percentiles", {})

    def _pctl(key: str) -> str:
        p = pctls.get(key)
        return f"{p:.0f}th" if p is not None else "—"

    lines += [
        "## Key Metrics",
        "",
        "_Percentile = where this value sits within the sector peer universe_",
        "",
        "| Metric | Value | Peer Percentile |",
        "|--------|-------|-----------------|",
        f"| Current Price | {_fmt_val(metrics.get('current_price'))} | — |",
        f"| P/E | {_fmt_val(metrics.get('pe_ratio'))} | {_pctl('pe_ratio')} |",
        f"| EV/EBITDA | {_fmt_val(metrics.get('ev_ebitda'))} | {_pctl('ev_ebitda')} |",
        f"| P/S | {_fmt_val(metrics.get('price_to_sales'))} | {_pctl('price_to_sales')} |",
        f"| P/FCF | {_fmt_val(metrics.get('price_to_fcf'))} | {_pctl('price_to_fcf')} |",
        f"| Gross Margin | {_fmt_pct(metrics.get('gross_margin'))} | {_pctl('gross_margin')} |",
        f"| Operating Margin | {_fmt_pct(metrics.get('operating_margin'))} | {_pctl('operating_margin')} |",
        f"| Net Margin | {_fmt_pct(metrics.get('net_margin'))} | {_pctl('net_margin')} |",
        f"| ROE | {_fmt_pct(metrics.get('roe'))} | {_pctl('roe')} |",
        f"| Debt/Equity | {_fmt_val(metrics.get('debt_to_equity'))} | {_pctl('debt_to_equity')} |",
        f"| Revenue YoY | {_fmt_pct(metrics.get('revenue_yoy'))} | {_pctl('revenue_yoy')} |",
        f"| Revenue CAGR 3Y | {_fmt_pct(metrics.get('revenue_cagr_3y'))} | {_pctl('revenue_cagr_3y')} |",
        f"| Return 3M | {_fmt_pct(metrics.get('return_3m'))} | {_pctl('return_3m')} |",
        f"| Return 12M | {_fmt_pct(metrics.get('return_12m'))} | {_pctl('return_12m')} |",
        "",
        "---",
        "",
    ]

    # Trends
    trends = metrics.get("trends", {})
    annual = trends.get("annual", [])
    quarterly = trends.get("quarterly", [])
    price_trend = trends.get("price", {})

    if annual or quarterly or price_trend:
        lines += ["## Trends", ""]

    if annual:
        lines += [
            "### Annual Performance",
            "",
            "| FY End | Revenue | Net Income | Net Margin | Rev YoY | NI YoY |",
            "|--------|---------|------------|------------|---------|--------|",
        ]
        for row in annual:
            lines.append(
                f"| {row['fiscal_year_end']} "
                f"| {_fmt_money(row['revenue'])} "
                f"| {_fmt_money(row['net_income'])} "
                f"| {_fmt_pct(row['net_margin'])} "
                f"| {_fmt_pct_arrow(row['revenue_yoy'])} "
                f"| {_fmt_pct_arrow(row['net_income_yoy'])} |"
            )
        lines += [""]

    if quarterly:
        lines += [
            "### Quarterly Performance",
            "",
            "| Quarter End | Revenue | Net Income | Net Margin |",
            "|-------------|---------|------------|------------|",
        ]
        for row in quarterly:
            lines.append(
                f"| {row['quarter_end']} "
                f"| {_fmt_money(row['revenue'])} "
                f"| {_fmt_money(row['net_income'])} "
                f"| {_fmt_pct(row['net_margin'])} |"
            )
        lines += [""]

    if price_trend:
        rets = price_trend.get("returns", {})
        lines += [
            "### Price Trend",
            "",
            "| Window | Return |",
            "|--------|--------|",
            f"| 1 month | {_fmt_pct_arrow(rets.get('1m'))} |",
            f"| 3 months | {_fmt_pct_arrow(rets.get('3m'))} |",
            f"| 6 months | {_fmt_pct_arrow(rets.get('6m'))} |",
            f"| 12 months | {_fmt_pct_arrow(rets.get('12m'))} |",
            "",
            f"Price vs 50-day MA: {_fmt_pct_arrow(price_trend.get('price_vs_ma50'))} | "
            f"vs 200-day MA: {_fmt_pct_arrow(price_trend.get('price_vs_ma200'))}",
            "",
        ]

    # Insider activity detail (trailing 6 months)
    buy_ct = metrics.get("insider_buy_count", 0) or 0
    sell_ct = metrics.get("insider_sell_count", 0) or 0
    if buy_ct + sell_ct > 0:
        lines += [
            "### Insider Activity (trailing 6 months)",
            "",
            f"- Buys: {buy_ct} transactions, {_fmt_money(metrics.get('insider_buy_value'))}",
            f"- Sells: {sell_ct} transactions, {_fmt_money(metrics.get('insider_sell_value'))}",
            f"- Net: {_fmt_money(metrics.get('insider_net_value'))}"
            + (" (cluster buying detected)" if metrics.get("insider_cluster") else ""),
            "",
        ]

    lines += ["---", ""]

    # Market positioning (unscored context: analyst consensus + short interest)
    ctx = metrics.get("market_context", {})
    analyst = ctx.get("analyst_consensus", {})
    short = ctx.get("short_interest", {})
    has_analyst = analyst.get("target_mean") is not None
    has_short = short.get("pct_of_float") is not None

    if has_analyst or has_short:
        lines += [
            "## Market Positioning",
            "",
            "_Unscored context — interpreted in synthesis, not part of the factor score._",
            "",
        ]

    if has_analyst:
        rec = analyst.get("recommendation")
        rec_str = rec.replace("_", " ").title() if rec else "N/A"
        lines += [
            "### Analyst Consensus",
            "",
            f"- Rating: **{rec_str}** "
            f"(mean {_fmt_val(analyst.get('recommendation_mean'))} on a 1=Strong Buy → 5=Sell scale, "
            f"{analyst.get('num_analysts', 'N/A')} analysts)",
            f"- Price target: {_fmt_val(analyst.get('target_mean'))} mean "
            f"({_fmt_val(analyst.get('target_low'))}–{_fmt_val(analyst.get('target_high'))} range)",
            f"- Implied upside to mean target: {_fmt_pct_arrow(analyst.get('upside_to_target'))}",
            "",
        ]

    if has_short:
        mom = short.get("mom_change")
        mom_note = ""
        if mom is not None:
            direction = "rising" if mom > 0.02 else ("falling" if mom < -0.02 else "flat")
            mom_note = f" ({direction} {_fmt_pct_arrow(mom)} vs prior month)"
        lines += [
            "### Short Interest",
            "",
            f"- Short % of float: {_fmt_pct(short.get('pct_of_float'))}{mom_note}",
            f"- Days to cover: {_fmt_val(short.get('days_to_cover'))}",
            "",
        ]

    if has_analyst or has_short:
        lines += ["---", ""]

    # Business description
    if description:
        lines += ["## Business Description", "", description[:500], "", "---", ""]

    # Synthesis
    if synthesis:
        lines += [
            "## Analysis",
            "",
            synthesis.get("narrative_md", ""),
            "",
            f"**Investment lean:** {synthesis.get('buy_sell_lean', 'N/A')} "
            f"(confidence: {synthesis.get('lean_confidence', 'N/A')})",
            "",
            "### Thesis",
            synthesis.get("thesis", ""),
            "",
            "### Bull Case",
            synthesis.get("bull_case", ""),
            "",
            "### Bear Case",
            synthesis.get("bear_case", ""),
            "",
            "### What Would Change the Picture",
            synthesis.get("what_would_change_picture", ""),
            "",
        ]
        caveats = synthesis.get("data_quality_caveats", [])
        if caveats:
            lines += ["### Data Quality Caveats", ""]
            for c in caveats:
                lines.append(f"- {c}")
            lines.append("")
    else:
        lines += ["## Analysis", "", "_LLM synthesis not available (no API key configured)._", ""]

    # Qualitative extractions (summary)
    if extractions:
        lines += ["## Qualitative Extractions", ""]
        for source, data in extractions.items():
            if isinstance(data, dict) and not data.get("unavailable"):
                score = data.get("score", "N/A")
                rationale = data.get("rationale", "")
                confidence = data.get("confidence", "N/A")
                lines.append(f"**{source.replace('_', ' ').title()}** — "
                              f"Score: {score}, Confidence: {confidence}")
                lines.append(f"> {rationale}")
                lines.append("")

    # Errors
    if errors:
        lines += ["## Source Availability", ""]
        for e in errors:
            lines.append(f"- ⚠️ {e}")
        lines.append("")

    return "\n".join(lines)


def _render_comparative_md(comparative: dict) -> str:
    lines = [
        "## Comparative Analysis",
        "",
        f"**Overall ranking:** {' > '.join(comparative.get('overall_ranking', []))}",
        "",
        comparative.get("ranking_rationale", ""),
        "",
        "### Dimension Rankings",
        "",
    ]
    for dim, data in comparative.get("dimension_rankings", {}).items():
        ranking = data.get("ranking", [])
        rationale = data.get("rationale", "")
        lines.append(f"**{dim.replace('_', ' ').title()}:** {' > '.join(ranking)}")
        lines.append(f"> {rationale}")
        lines.append("")
    return "\n".join(lines)


def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def _fmt_pct_arrow(v) -> str:
    if v is None:
        return "N/A"
    arrow = "▲" if v > 0.005 else ("▼" if v < -0.005 else "→")
    return f"{arrow} {v * 100:+.1f}%"


def _fmt_money(v) -> str:
    if v is None:
        return "N/A"
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1e12:
        return f"{sign}${av / 1e12:.2f}T"
    if av >= 1e9:
        return f"{sign}${av / 1e9:.2f}B"
    if av >= 1e6:
        return f"{sign}${av / 1e6:.1f}M"
    return f"{sign}${av:,.0f}"


def _fmt_val(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}"

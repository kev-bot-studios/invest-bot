# Invest Bot — Equity Analysis Pipeline

A personal research tool that collects free investment data for one or more tickers, computes peer-relative quantitative factor scores, optionally runs LLM-based interpretation of qualitative sources (SEC filings, news), and produces a per-ticker markdown report with 1–5 factor scores.

> **Disclaimer:** This is a research aid, not investment advice. Scores and summaries are heuristics over public data and LLM output.

## Quick start

```bash
pip3 install -r requirements.txt
python3 -m src.cli --tickers AAPL MSFT --sector technology  # tickers + peer universe
python3 -m src.cli --tickers NVDA AMD --sector technology   # any tickers
python3 -m src.cli --tickers XOM CVX --sector energy        # rank against energy peers
python3 -m src.cli --tickers AAPL --sector technology --llm-provider ollama  # force a specific LLM
python3 -m src.cli --tickers JPM --sector financials --no-llm  # computed scores only, no API keys
python3 -m src.cli --scan-sector --sector technology         # screen the whole sector, deep-dive the standouts
```

`--sector` is **required** (or set `sector:` in the config) — there is no default. Scores are
peer-relative, so the sector must fit the tickers; running with a missing or unknown sector
fails fast and lists the valid options. See [Available sectors](#available-sectors).

Reports are written to `reports/`, and every run is persisted to SQLite (`invest_bot.db`) for auditability.

### Sector scan (idea generation)

Instead of bringing your own tickers, `--scan-sector` brings names to you. It runs in two passes:

1. **Wide pass** — scores *every* name in the sector peer universe on the computed factors only (no LLM calls), so it's cheap enough to run often.
2. **Deep pass** — flags the most **value-divergent** names (high quality + growth, low valuation — the "why is this cheap?" candidates) and runs the full LLM analysis only on those.

```bash
python3 -m src.cli --scan-sector --sector technology              # default: deep-dive top 3
python3 -m src.cli --scan-sector --sector energy --deep-count 2   # deep-dive top 2
python3 -m src.cli --scan-sector --sector financials --no-llm     # wide screen only, no deep pass
```

Output is a single combined report (`reports/<sector>_scan_<timestamp>_<id>.md`): a ranked screen table of the whole universe (★ marks the flagged names), followed by the full deep dive on each flagged name and a head-to-head comparative ranking. `--deep-count N` sets how many names get the deep pass (default 3); flagged names that clear the divergence bar (quality & growth ≥ 3.5, valuation ≤ 2.5) come first, and if fewer than `N` qualify the closest candidates by divergence fill the rest. The first scan of a sector is slow (it pulls every peer fresh); subsequent runs use the cache.

### Available sectors

`--sector` (or `sector:` in the config) selects which peer list to rank against, from
`config/peers/<sector>.yaml`. It is **required** — there is no default sector. Scores are
peer-relative, so match the sector to the tickers:

| `--sector` value | Peer universe |
|------------------|---------------|
| `technology` | ~25 large-cap tech names |
| `financials` | Banks, payments, insurers, asset managers |
| `healthcare` | Pharma, biotech, devices, payers |
| `energy` | Oil & gas majors, E&P, services, midstream |
| `consumer_discretionary` | Retail, autos, restaurants, travel |
| `consumer_staples` | Food, beverage, household, tobacco |

Add your own by dropping a `config/peers/<name>.yaml` with a `sector:` field and a `tickers:` list.

## What it produces

- **Factor scorecard** (1–5, peer-relative quintiles): Valuation, Quality, Growth, Momentum, Insider Activity — computed deterministically in Python — plus Filing Tone and News Tone (LLM-judged, labeled separately).
- **Trend tables**: 6 years of annual revenue/income/margins with YoY changes, 8 quarters of quarterly performance, price returns across 1m/3m/6m/12m windows, and insider buy/sell activity from SEC Form 4.
- **Peer percentile context** for every key metric against a ~25-ticker sector universe.
- **LLM synthesis** (when an API key is configured): thesis, bull/bear case, buy/sell lean with confidence, data-quality caveats, plus a head-to-head comparative ranking for multi-ticker runs.

## How the factor scorecard is computed

All scores are on a **1–5 scale** and fall into two families that are kept strictly
separate (and labeled as such in the report): **computed** factors, which are deterministic
and peer-relative, and **LLM-judged** factors, which are qualitative.

### Computed factors (deterministic, peer-relative)

Each underlying metric is ranked against the **peer universe** (a ~25-ticker sector list in
`config/peers/<sector>.yaml`) rather than scored on an absolute basis. The mechanics:

1. **Compute the raw metric** in Python (e.g. P/E, ROE, 6-month return). The LLM never does this.
2. **Find its percentile** within the peer distribution for that metric.
3. **Map percentile → 1–5 quintile**, respecting direction:
   - *Higher-is-better* metrics (margins, returns, growth): top quintile (≥80th pct) = **5**, bottom (<20th) = **1**.
   - *Lower-is-better* metrics (P/E, EV/EBITDA, debt/EBITDA): the scale inverts — cheapest quintile (≤20th pct) = **5**.
4. **Average the metric scores within each factor** (equal-weighted) to get the factor score.
   Metrics with no data are skipped, not counted as zero.

| Factor | Underlying metrics (equal-weighted) | Direction |
|--------|--------------------------------------|-----------|
| **Valuation** | P/E, EV/EBITDA, P/S, P/FCF | lower = better |
| **Quality** | gross / operating / net margin, ROE, ROIC, FCF margin, debt/EBITDA | higher = better (debt: lower) |
| **Growth** | revenue YoY, revenue CAGR 3y, EPS CAGR 3y, FCF growth YoY | higher = better |
| **Momentum** | 3m / 6m / 12m return, price vs 52-week high | higher = better |
| **Insider** | net Form 4 buy/sell value & count, trailing 6m | see rule below |

**Negative-earnings fallback:** when a company has negative earnings, P/E and EV/EBITDA are
meaningless, so Valuation falls back to P/S, EV/Revenue, and P/FCF, and the report flags it.

**Insider rule (not a percentile):** insider activity is scored by logic, not peer ranking:
no activity = **3** (neutral); net buying = **4**, or **5** if cluster buying (≥3 buys) is
detected; net selling is scaled by size relative to market cap — routine comp-plan selling
(<0.01% of mcap) stays neutral at **3**, moderate selling = **2**, heavy selling (>0.1%) = **1**.

### LLM-judged factors (qualitative)

These appear in the scorecard only when an LLM key is configured, and are always labeled
separately because they carry more noise than the computed scores:

- **Filing Tone** — average of two year-over-year *delta* analyses of the latest 10-K: MD&A
  tone shift and risk-factor diff. Measures what *changed* (hedging, new/removed risks), never
  absolute sentiment (filings are lawyered to neutrality).
- **News Tone** — read of trailing-30-day headlines against a fixed 1–5 rubric.

### Composite

The composite is a **weighted average of whichever factors are present**, using the weights
in `config/run_config.yaml` (defaults: Quality 0.25, Growth 0.25, Valuation 0.20,
Momentum 0.15, Insider 0.05, Filing Tone 0.05, News Tone 0.05). Weights are renormalized
over the available factors, so a run with no LLM scores still produces a valid composite from
the computed factors alone. It is always shown *alongside* the factor breakdown, never instead of it.

> Scores are peer-relative heuristics, not absolute quality judgments — a "5" means top-quintile
> *within the configured peer set*, so the peer list materially shapes every score.

## Data sources (all free)

| Source | Provides | Key required |
|--------|----------|--------------|
| yfinance | Prices, fundamentals, news | No |
| SEC EDGAR | XBRL fundamentals, 10-K MD&A + risk factors, Form 4 insider trades | No |
| FRED | Macro series (rates, sector indicators) | `FRED_API_KEY` |
| Finnhub (optional) | Analyst recommendations, earnings surprises | `FINNHUB_API_KEY` |

## LLM providers

Configured in `config/run_config.yaml` with ordered fallback — the first provider with
a usable key (or, for local Ollama, a reachable server) wins; on rate-limit or error the
next is tried. Defaults:

1. Google AI Studio (Gemini 2.5 Flash) — `GOOGLE_AI_STUDIO_API_KEY`
2. OpenRouter free models — `OPENROUTER_API_KEY`
3. Ollama (local) — no key; runs whatever model you've pulled (default `llama3.2:latest`)

Without any key the pipeline still runs and produces all computed scores.

**Pick a provider for a run** with `--llm-provider` — it's moved to the front of the chain
(the others stay as fallbacks), so this both *prioritizes* a provider and is the way to force
the local model:

```bash
python3 -m src.cli --tickers AAPL --llm-provider ollama          # prefer local Llama
python3 -m src.cli --tickers AAPL --llm-provider google_ai_studio
```

The provider/model actually used is printed in the run header and recorded at the top of each
markdown report (e.g. _"generated by: ollama / llama3.2:latest"_), so you can always tell
whether a report's LLM-judged scores came from a cloud model or your local one.

### Local fallback via Ollama

The last provider in the default chain is a **local** [Ollama](https://ollama.com) server, so
the LLM layer works with no cloud key and no network. It needs no `api_key_env` — a provider
without one is treated as keyless and assumed reachable, falling through gracefully if the
server isn't running.

```bash
ollama serve                 # start the server (usually already running)
ollama pull llama3.2         # or any model; list what you have with `ollama list`
```

Then set the `model:` under the `ollama` provider in `config/run_config.yaml` to match a
model you've pulled. Small local models are noisier than the cloud providers, so treat their
LLM-judged scores (Filing Tone, News Tone) and synthesis as a rough estimate.

## Caching & data freshness

Raw API responses are cached to `cache/` with a per-source TTL, so reruns don't re-hit
the network. Each TTL is matched to how fast that data actually changes:

| Data | Source | TTL | Rationale |
|------|--------|-----|-----------|
| Prices (OHLCV) | yfinance | 1 day | Daily bars |
| Fundamentals (info, statements) | yfinance | ~90 days (1 quarter) | Only changes at earnings |
| News headlines | yfinance | 1 day | Fresh without hammering the endpoint |
| Company facts (XBRL) | EDGAR | ~90 days | Tied to filing cadence |
| Filing sections (MD&A / risk) | EDGAR | ~90 days | Filings are immutable once published |
| Form 4 insider | EDGAR | 7 days | New filings trickle in weekly |
| FRED macro | FRED | 7 days | Most series are weekly/monthly |
| Finnhub analyst | Finnhub | 7 days | Consensus moves slowly |

The check is a hard cutoff: on each request the cache compares the entry's age to its
TTL and re-fetches only if it's stale. Note this means a run "today" can still serve
fundamentals fetched up to a quarter ago — `tickers.fetched_at` records when the run
**assembled** the snapshot, not when each source was originally pulled.

- **Force a full refresh:** `rm -rf cache/` — the next run re-fetches everything.
- **Tune a threshold:** edit the `_TTL_H` constant in the relevant `src/data/*_source.py`.

## Persistence

Every run is written to SQLite (`invest_bot.db`); the markdown reports in `reports/` are a
human-readable mirror of the `outputs` table. Both are gitignored (local-only). Key tables:

- `runs` — one row per run (id, `started_at`, config, status)
- `tickers` — full raw snapshot, computed metrics, scores, `price_at_run`, `fetched_at` (UTC)
- `outputs` — rendered markdown per ticker
- `llm_calls` — prompt hash, model, tokens, latency (populated once an LLM key is set)
- `peer_universe` — per-quarter peer distributions

JSON columns (`snapshot_json`, `metrics_json`, `scores_json`) are queryable with
`json_extract(col, '$.key')`, e.g.:

```bash
sqlite3 -header -column invest_bot.db \
  "SELECT ticker, fetched_at, json_extract(scores_json,'\$.composite') AS composite
   FROM tickers ORDER BY ticker;"
```

## Layout

```
src/
  config.py            YAML run config + provider/weights config
  db.py                SQLite persistence (runs, snapshots, scores, llm_calls)
  pipeline.py          Orchestration + report rendering
  cli.py               CLI entry point (rich scorecard output)
  data/                yfinance, EDGAR, FRED, Finnhub sources + TTL cache
  metrics/             Pure-Python metrics engine + peer universe
  llm/                 Provider-fallback client, versioned prompts, extraction, synthesis
config/
  run_config.yaml      Tickers, sources, providers, factor weights
  peers/technology.yaml  Sector peer list for percentile ranking
```

## Design notes

- The LLM never does arithmetic; every ratio/percentile/score is computed in Python.
- Filing analysis is **tone delta** (year-over-year change), never absolute sentiment.
- Per-source TTL caching (see [Caching & data freshness](#caching--data-freshness)); deep-dive fetches run only for focus tickers.
- Any single source failing degrades gracefully — the report lists unavailable inputs.

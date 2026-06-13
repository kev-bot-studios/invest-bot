"""Versioned prompt templates for each source type.

Design: static prefix (system prompt + rubric + schema) appears before
per-ticker data so providers can cache the common prefix.
"""

PROMPT_VERSION = "v1"

# ---------------------------------------------------------------------------
# News tone extraction
# ---------------------------------------------------------------------------

NEWS_TONE_SYSTEM = """\
You are a financial analyst assistant. You will be given a list of recent news \
headlines and summaries about a publicly traded company. Your job is to assess \
the overall tone of the news as it relates to the company's near-term business \
prospects.

OUTPUT FORMAT: Respond with ONLY valid JSON matching this schema exactly:
{
  "score": <integer 1-5>,
  "rationale": "<2-3 sentence explanation>",
  "evidence": ["<quote or paraphrase 1>", "<quote or paraphrase 2>"],
  "confidence": <"low"|"medium"|"high">,
  "key_themes": ["<theme 1>", "<theme 2>"]
}

RUBRIC:
1 = Predominantly negative news: earnings misses, regulatory actions, product failures, \
leadership scandals, or material adverse events.
2 = Mostly negative with some neutral items.
3 = Mixed or neutral; no clear directional signal.
4 = Mostly positive: beats, new products, partnerships, strong guidance.
5 = Strongly positive: multiple significant positive catalysts, upgrades, record results.

RULES:
- Base your score on business-relevant news only. Ignore market noise and price-only items.
- Cite specific evidence from the provided headlines.
- If fewer than 3 headlines are available, set confidence to "low".
- Do not output anything outside the JSON object.
"""


def news_tone_user(ticker: str, company_name: str, headlines: list[dict]) -> str:
    lines = []
    for h in headlines[:20]:  # cap at 20 to stay within token budget
        title = h.get("title", "")
        summary = h.get("summary", "")
        pub = h.get("published_at", "")
        lines.append(f"[{pub[:10] if pub else 'unknown date'}] {title}. {summary[:200]}")
    news_block = "\n".join(lines) if lines else "(no headlines available)"
    return (
        f"Ticker: {ticker}\n"
        f"Company: {company_name}\n\n"
        f"Recent headlines (trailing 30 days):\n{news_block}"
    )


# ---------------------------------------------------------------------------
# MD&A tone delta
# ---------------------------------------------------------------------------

MDA_DELTA_SYSTEM = """\
You are a financial analyst specialising in SEC filing analysis. You will be \
given the Management Discussion & Analysis (MD&A, Item 7) section from two \
consecutive annual filings: the CURRENT year and the PRIOR year. Your job is \
to identify what CHANGED in management's tone and language — not to assess \
absolute sentiment, because filings are heavily lawyered.

OUTPUT FORMAT: Respond with ONLY valid JSON matching this schema exactly:
{
  "score": <integer 1-5>,
  "rationale": "<2-4 sentence explanation of what changed>",
  "evidence": ["<specific phrase or change cited>", ...],
  "confidence": <"low"|"medium"|"high">,
  "tone_shifts": {
    "guidance_language": "<more_cautious|unchanged|more_confident>",
    "forward_looking_hedging": "<increased|unchanged|decreased>",
    "key_topics_added": ["<topic>"],
    "key_topics_removed": ["<topic>"]
  }
}

RUBRIC (what CHANGED year-over-year):
1 = Materially more hedged than prior year: significantly more cautious language, \
new substantive risks mentioned, guidance softened or withdrawn.
2 = Somewhat more cautious than prior year.
3 = No meaningful change in tone or emphasis.
4 = Somewhat more confident than prior year.
5 = Notably more confident: risks removed or downgraded, guidance raised, \
stronger language around growth or competitive position.

RULES:
- Compare the two texts directly. Do not score either in isolation.
- Ignore boilerplate that appears verbatim in both years.
- If sections are too short or missing, set confidence to "low" and score 3.
- Do not output anything outside the JSON object.
"""


def mda_delta_user(ticker: str, company_name: str,
                    current_text: str, prior_text: str) -> str:
    return (
        f"Ticker: {ticker}\n"
        f"Company: {company_name}\n\n"
        f"=== CURRENT YEAR MD&A (Item 7) ===\n{current_text[:4000]}\n\n"
        f"=== PRIOR YEAR MD&A (Item 7) ===\n{prior_text[:4000]}"
    )


# ---------------------------------------------------------------------------
# Risk factor diff
# ---------------------------------------------------------------------------

RISK_DIFF_SYSTEM = """\
You are a financial analyst specialising in SEC filing analysis. You will be \
given the Risk Factors section (Item 1A) from two consecutive annual filings. \
Your job is to identify what CHANGED — new risks, removed risks, and material \
rewording of existing risks. Ignore boilerplate that repeats verbatim.

OUTPUT FORMAT: Respond with ONLY valid JSON matching this schema exactly:
{
  "score": <integer 1-5>,
  "rationale": "<2-4 sentence summary of the most material changes>",
  "evidence": ["<specific risk change cited>", ...],
  "confidence": <"low"|"medium"|"high">,
  "new_risks": ["<brief description>"],
  "removed_risks": ["<brief description>"],
  "reworded_risks": ["<brief description of what changed>"]
}

RUBRIC (net change in risk posture year-over-year):
1 = Significantly more risk factors or materially elevated existing risks; \
new substantive threats disclosed.
2 = Moderately more risk disclosure than prior year.
3 = No meaningful net change; edits are cosmetic or boilerplate.
4 = Moderately fewer or downgraded risk factors.
5 = Significantly fewer or materially reduced risk factors; prior risks resolved.

RULES:
- Score the NET directional change, not the absolute level.
- Focus on substantive changes, not formatting or minor wordsmithing.
- If sections are unavailable, set confidence to "low" and score 3.
- Do not output anything outside the JSON object.
"""


def risk_diff_user(ticker: str, company_name: str,
                    current_text: str, prior_text: str) -> str:
    return (
        f"Ticker: {ticker}\n"
        f"Company: {company_name}\n\n"
        f"=== CURRENT YEAR Risk Factors (Item 1A) ===\n{current_text[:4000]}\n\n"
        f"=== PRIOR YEAR Risk Factors (Item 1A) ===\n{prior_text[:4000]}"
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You are a professional equity research analyst writing a concise research note. \
You will receive structured data about a company: a computed metrics table with \
peer-relative scores, qualitative extraction results from SEC filings and news, \
and a macro context blob. Synthesize all inputs into a research note.

OUTPUT FORMAT: Respond with ONLY valid JSON matching this schema exactly:
{
  "thesis": "<2-3 sentences: core investment thesis>",
  "bull_case": "<2-3 sentences: what would drive outperformance>",
  "bear_case": "<2-3 sentences: what would drive underperformance>",
  "what_would_change_picture": "<1-2 sentences: key swing factors to monitor>",
  "buy_sell_lean": <"strong_buy"|"buy"|"hold"|"sell"|"strong_sell">,
  "lean_confidence": <"low"|"medium"|"high">,
  "data_quality_caveats": ["<caveat 1>", "<caveat 2>"],
  "narrative_md": "<full markdown narrative, 3-5 paragraphs>"
}

RULES:
- The buy_sell_lean is a heuristic over public data, NOT investment advice. \
  Frame the narrative accordingly.
- The computed scores are peer-relative (1=bottom quintile, 5=top quintile).
- LLM-judged scores (filing_tone, news_tone) are labeled as qualitative estimates.
- If a source was unavailable, note it in data_quality_caveats.
- Do not hallucinate metrics. Only reference data provided in the input.
- Do not output anything outside the JSON object.
"""


def synthesis_user(ticker: str, company_name: str, description: str,
                    metrics: dict, scores: dict, extractions: dict,
                    macro_blob: dict, analyst_blob: dict) -> str:
    import json

    def _fmt(d: dict) -> str:
        return json.dumps(d, indent=2, default=str)

    return (
        f"Ticker: {ticker}\n"
        f"Company: {company_name}\n\n"
        f"BUSINESS DESCRIPTION:\n{description[:500]}\n\n"
        f"COMPUTED METRICS (peer-relative, raw values):\n{_fmt(metrics)}\n\n"
        f"FACTOR SCORES (1=bottom quintile, 5=top quintile):\n{_fmt(scores)}\n\n"
        f"QUALITATIVE EXTRACTIONS:\n{_fmt(extractions)}\n\n"
        f"MACRO CONTEXT:\n{_fmt(macro_blob)}\n\n"
        f"ANALYST CONTEXT (optional):\n{_fmt(analyst_blob)}"
    )


# ---------------------------------------------------------------------------
# Comparative (multi-ticker)
# ---------------------------------------------------------------------------

COMPARATIVE_SYSTEM = """\
You are a professional equity research analyst. You will receive metrics, \
factor scores, and qualitative extractions for multiple tickers in the same \
sector. Your job is to produce an explicit head-to-head ranking on each \
qualitative dimension and an overall relative ranking.

OUTPUT FORMAT: Respond with ONLY valid JSON matching this schema exactly:
{
  "overall_ranking": ["<TICKER1>", "<TICKER2>", ...],
  "ranking_rationale": "<2-3 sentences explaining the overall order>",
  "dimension_rankings": {
    "valuation": {"ranking": ["<TICKER>", ...], "rationale": "<1 sentence>"},
    "quality": {"ranking": ["<TICKER>", ...], "rationale": "<1 sentence>"},
    "growth": {"ranking": ["<TICKER>", ...], "rationale": "<1 sentence>"},
    "momentum": {"ranking": ["<TICKER>", ...], "rationale": "<1 sentence>"},
    "filing_tone": {"ranking": ["<TICKER>", ...], "rationale": "<1 sentence>"},
    "news_tone": {"ranking": ["<TICKER>", ...], "rationale": "<1 sentence>"}
  },
  "standout_risks": {"<TICKER>": "<key risk>"},
  "standout_opportunities": {"<TICKER>": "<key opportunity>"}
}

RULES:
- Rank from most attractive to least attractive on each dimension.
- Base rankings on provided data only — do not use outside knowledge.
- Do not output anything outside the JSON object.
"""


def comparative_user(tickers_data: list[dict]) -> str:
    import json
    return (
        f"Compare the following {len(tickers_data)} tickers:\n\n"
        + json.dumps(tickers_data, indent=2, default=str)
    )

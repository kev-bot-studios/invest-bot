"""Per-source LLM extraction with pydantic validation and one retry."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from ..db import insert_llm_call
from .client import LLMClient, prompt_hash
from . import prompts as P


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class NewsToneResponse(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str
    evidence: list[str]
    confidence: str
    key_themes: list[str] = Field(default_factory=list)


class ToneShifts(BaseModel):
    guidance_language: str = ""
    forward_looking_hedging: str = ""
    key_topics_added: list[str] = Field(default_factory=list)
    key_topics_removed: list[str] = Field(default_factory=list)


class MdaDeltaResponse(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str
    evidence: list[str]
    confidence: str
    tone_shifts: ToneShifts = Field(default_factory=ToneShifts)


class RiskDiffResponse(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str
    evidence: list[str]
    confidence: str
    new_risks: list[str] = Field(default_factory=list)
    removed_risks: list[str] = Field(default_factory=list)
    reworded_risks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _call_and_validate(
    client: LLMClient,
    system: str,
    user: str,
    model_cls: type[BaseModel],
    source_type: str,
    ticker: str,
    run_id: str,
    db_path: str,
) -> Optional[BaseModel]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    phash = prompt_hash(messages)

    for attempt in range(2):
        try:
            t0 = time.time()
            raw, model_used, in_tok, out_tok = client.complete(
                messages,
                response_format={"type": "json_object"},
                # Thinking models (e.g. Gemini 2.5 Flash) spend hidden reasoning
                # tokens against this budget before emitting JSON; the large
                # filing-delta prompts need generous headroom or the JSON truncates.
                max_tokens=6000,
            )
            latency = (time.time() - t0) * 1000

            validated = model_cls.model_validate(raw)

            insert_llm_call(
                db_path=db_path,
                call_id=str(uuid.uuid4()),
                run_id=run_id,
                ticker=ticker,
                source_type=source_type,
                prompt_hash=phash,
                model=model_used,
                input_tokens=in_tok,
                output_tokens=out_tok,
                response=raw,
                latency_ms=latency,
                cost_estimate=0.0,
            )
            return validated

        except (ValidationError, ValueError, RuntimeError) as e:
            if attempt == 0:
                continue
            print(f"  [llm] {source_type} extraction failed for {ticker}: {e}")
            return None

    return None


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_news_tone(
    ticker: str,
    company_name: str,
    headlines: list[dict],
    client: LLMClient,
    run_id: str,
    db_path: str,
) -> Optional[dict]:
    if not headlines:
        return {"score": 3, "rationale": "No headlines available.", "evidence": [],
                "confidence": "low", "key_themes": []}

    user_msg = P.news_tone_user(ticker, company_name, headlines)
    result = _call_and_validate(
        client, P.NEWS_TONE_SYSTEM, user_msg,
        NewsToneResponse, "news_tone", ticker, run_id, db_path,
    )
    return result.model_dump() if result else None


def extract_mda_delta(
    ticker: str,
    company_name: str,
    current_sections: dict,
    prior_sections: dict,
    client: LLMClient,
    run_id: str,
    db_path: str,
) -> Optional[dict]:
    current_mda = current_sections.get("item_7", "")
    prior_mda = prior_sections.get("item_7", "")

    if not current_mda and not prior_mda:
        return {"score": 3, "rationale": "MD&A sections not available.", "evidence": [],
                "confidence": "low", "tone_shifts": {}}

    user_msg = P.mda_delta_user(ticker, company_name, current_mda, prior_mda)
    result = _call_and_validate(
        client, P.MDA_DELTA_SYSTEM, user_msg,
        MdaDeltaResponse, "mda_delta", ticker, run_id, db_path,
    )
    return result.model_dump() if result else None


def extract_risk_diff(
    ticker: str,
    company_name: str,
    current_sections: dict,
    prior_sections: dict,
    client: LLMClient,
    run_id: str,
    db_path: str,
) -> Optional[dict]:
    current_risks = current_sections.get("item_1a", "")
    prior_risks = prior_sections.get("item_1a", "")

    if not current_risks and not prior_risks:
        return {"score": 3, "rationale": "Risk factor sections not available.", "evidence": [],
                "confidence": "low", "new_risks": [], "removed_risks": [], "reworded_risks": []}

    user_msg = P.risk_diff_user(ticker, company_name, current_risks, prior_risks)
    result = _call_and_validate(
        client, P.RISK_DIFF_SYSTEM, user_msg,
        RiskDiffResponse, "risk_diff", ticker, run_id, db_path,
    )
    return result.model_dump() if result else None

"""Synthesis and comparative LLM calls."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from ..db import insert_llm_call
from .client import LLMClient, prompt_hash
from . import prompts as P


class SynthesisResponse(BaseModel):
    thesis: str
    bull_case: str
    bear_case: str
    what_would_change_picture: str
    buy_sell_lean: str
    lean_confidence: str
    data_quality_caveats: list[str] = Field(default_factory=list)
    narrative_md: str


class DimensionRanking(BaseModel):
    ranking: list[str]
    rationale: str


class ComparativeResponse(BaseModel):
    overall_ranking: list[str]
    ranking_rationale: str
    dimension_rankings: dict[str, DimensionRanking] = Field(default_factory=dict)
    standout_risks: dict[str, str] = Field(default_factory=dict)
    standout_opportunities: dict[str, str] = Field(default_factory=dict)


def synthesize_ticker(
    ticker: str,
    company_name: str,
    description: str,
    metrics: dict,
    scores: dict,
    extractions: dict,
    macro_blob: dict,
    analyst_blob: dict,
    client: LLMClient,
    run_id: str,
    db_path: str,
) -> Optional[dict]:
    messages = [
        {"role": "system", "content": P.SYNTHESIS_SYSTEM},
        {"role": "user", "content": P.synthesis_user(
            ticker, company_name, description,
            metrics, scores, extractions, macro_blob, analyst_blob,
        )},
    ]
    phash = prompt_hash(messages)

    for attempt in range(2):
        try:
            t0 = time.time()
            raw, model_used, in_tok, out_tok = client.complete(
                messages,
                response_format={"type": "json_object"},
                max_tokens=8000,  # narrative + thinking-token headroom (Gemini 2.5 Flash)
            )
            latency = (time.time() - t0) * 1000

            validated = SynthesisResponse.model_validate(raw)
            insert_llm_call(
                db_path=db_path,
                call_id=str(uuid.uuid4()),
                run_id=run_id,
                ticker=ticker,
                source_type="synthesis",
                prompt_hash=phash,
                model=model_used,
                input_tokens=in_tok,
                output_tokens=out_tok,
                response=raw,
                latency_ms=latency,
                cost_estimate=0.0,
            )
            return validated.model_dump()

        except (ValidationError, ValueError, RuntimeError) as e:
            if attempt == 0:
                continue
            print(f"  [llm] synthesis failed for {ticker}: {e}")
            return None

    return None


def compare_tickers(
    tickers_data: list[dict],
    client: LLMClient,
    run_id: str,
    db_path: str,
) -> Optional[dict]:
    if len(tickers_data) < 2:
        return None

    messages = [
        {"role": "system", "content": P.COMPARATIVE_SYSTEM},
        {"role": "user", "content": P.comparative_user(tickers_data)},
    ]
    phash = prompt_hash(messages)

    for attempt in range(2):
        try:
            t0 = time.time()
            raw, model_used, in_tok, out_tok = client.complete(
                messages,
                response_format={"type": "json_object"},
                max_tokens=6000,  # multi-ticker ranking + thinking-token headroom
            )
            latency = (time.time() - t0) * 1000

            validated = ComparativeResponse.model_validate(raw)
            insert_llm_call(
                db_path=db_path,
                call_id=str(uuid.uuid4()),
                run_id=run_id,
                ticker="COMPARATIVE",
                source_type="comparative",
                prompt_hash=phash,
                model=model_used,
                input_tokens=in_tok,
                output_tokens=out_tok,
                response=raw,
                latency_ms=latency,
                cost_estimate=0.0,
            )
            return validated.model_dump()

        except (ValidationError, ValueError, RuntimeError) as e:
            if attempt == 0:
                continue
            print(f"  [llm] comparative call failed: {e}")
            return None

    return None

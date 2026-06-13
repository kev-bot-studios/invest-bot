"""SQLite persistence layer — schema init and CRUD helpers."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    config_json TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    provider_used  TEXT,
    status      TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS tickers (
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    ticker        TEXT NOT NULL,
    sector        TEXT,
    snapshot_json TEXT,
    metrics_json  TEXT,
    scores_json   TEXT,
    price_at_run  REAL,
    fetched_at    TEXT,            -- UTC ISO-8601: when this ticker's snapshot was assembled
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    ticker        TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    prompt_hash   TEXT,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    response_json TEXT,
    latency_ms    REAL,
    cost_estimate REAL
);

CREATE TABLE IF NOT EXISTS peer_universe (
    sector            TEXT NOT NULL,
    as_of_quarter     TEXT NOT NULL,
    tickers_json      TEXT NOT NULL,
    distributions_json TEXT NOT NULL,
    PRIMARY KEY (sector, as_of_quarter)
);

CREATE TABLE IF NOT EXISTS outputs (
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    ticker         TEXT NOT NULL,
    synthesis_md   TEXT,
    comparative_md TEXT,
    PRIMARY KEY (run_id, ticker)
);
"""


@contextmanager
def _conn(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path) as con:
        con.executescript(SCHEMA)


def insert_run(db_path: str, run_id: str, started_at: str, config: dict,
               prompt_version: str) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, config_json, prompt_version, status) "
            "VALUES (?, ?, ?, ?, 'running')",
            (run_id, started_at, json.dumps(config), prompt_version),
        )


def update_run_status(db_path: str, run_id: str, status: str,
                      provider_used: Optional[str] = None) -> None:
    with _conn(db_path) as con:
        con.execute(
            "UPDATE runs SET status=?, provider_used=? WHERE run_id=?",
            (status, provider_used, run_id),
        )


def upsert_ticker(db_path: str, run_id: str, ticker: str, sector: Optional[str],
                  snapshot: Optional[dict], metrics: Optional[dict],
                  scores: Optional[dict], price: Optional[float],
                  fetched_at: Optional[str] = None) -> None:
    with _conn(db_path) as con:
        con.execute(
            """INSERT INTO tickers (run_id, ticker, sector, snapshot_json, metrics_json,
               scores_json, price_at_run, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, ticker) DO UPDATE SET
                 sector=excluded.sector,
                 snapshot_json=excluded.snapshot_json,
                 metrics_json=excluded.metrics_json,
                 scores_json=excluded.scores_json,
                 price_at_run=excluded.price_at_run,
                 fetched_at=excluded.fetched_at""",
            (
                run_id, ticker, sector,
                json.dumps(snapshot) if snapshot else None,
                json.dumps(metrics) if metrics else None,
                json.dumps(scores) if scores else None,
                price,
                fetched_at,
            ),
        )


def insert_llm_call(db_path: str, call_id: str, run_id: str, ticker: str,
                    source_type: str, prompt_hash: str, model: str,
                    input_tokens: int, output_tokens: int, response: dict,
                    latency_ms: float, cost_estimate: float) -> None:
    with _conn(db_path) as con:
        con.execute(
            """INSERT INTO llm_calls
               (call_id, run_id, ticker, source_type, prompt_hash, model,
                input_tokens, output_tokens, response_json, latency_ms, cost_estimate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_id, run_id, ticker, source_type, prompt_hash, model,
             input_tokens, output_tokens, json.dumps(response), latency_ms, cost_estimate),
        )


def upsert_peer_universe(db_path: str, sector: str, as_of_quarter: str,
                         tickers: list[str], distributions: dict) -> None:
    with _conn(db_path) as con:
        con.execute(
            """INSERT INTO peer_universe (sector, as_of_quarter, tickers_json, distributions_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(sector, as_of_quarter) DO UPDATE SET
                 tickers_json=excluded.tickers_json,
                 distributions_json=excluded.distributions_json""",
            (sector, as_of_quarter, json.dumps(tickers), json.dumps(distributions)),
        )


def get_peer_universe(db_path: str, sector: str,
                      as_of_quarter: str) -> Optional[dict]:
    with _conn(db_path) as con:
        row = con.execute(
            "SELECT distributions_json FROM peer_universe WHERE sector=? AND as_of_quarter=?",
            (sector, as_of_quarter),
        ).fetchone()
    if row:
        return json.loads(row["distributions_json"])
    return None


def upsert_output(db_path: str, run_id: str, ticker: str,
                  synthesis_md: Optional[str], comparative_md: Optional[str] = None) -> None:
    with _conn(db_path) as con:
        con.execute(
            """INSERT INTO outputs (run_id, ticker, synthesis_md, comparative_md)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(run_id, ticker) DO UPDATE SET
                 synthesis_md=excluded.synthesis_md,
                 comparative_md=excluded.comparative_md""",
            (run_id, ticker, synthesis_md, comparative_md),
        )


def get_run_tickers(db_path: str, run_id: str) -> list[dict[str, Any]]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM tickers WHERE run_id=?", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]

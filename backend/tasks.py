"""Celery task definitions for AuditPilot.

A single task, :func:`run_due_diligence`, drives the whole pipeline and reports
progress to Redis so the FastAPI layer (and Streamlit) can poll it:

    0%   Fetching SEC filing
    20%  Embedding into ChromaDB
    35%  Running Planner
    55%  Running Risk Analyst
    70%  Running Critic (+ possible loop)
    85%  Writing memo
    100% Complete

Redis keys (all TTL'd):
    progress:{task_id}  -> JSON {percent, stage, status, ticker, year, updated_at, error}
    stream:{task_id}    -> incrementally appended memo text (for live SSE)
    memo:{task_id}      -> final markdown memo

Redis is used as Celery's broker and result backend.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis
from celery import Celery

from backend.agents.graph import NODE_PROGRESS, compiled_graph
from backend.config import settings
from backend.tools import edgar, vector_store

KEY_TTL = 60 * 60 * 24  # 24h

celery_app = Celery(
    "auditpilot",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_track_started=True,
    result_expires=KEY_TTL,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)


def _redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)


def set_progress(
    r: redis.Redis,
    task_id: str,
    percent: int,
    stage: str,
    *,
    status: str = "running",
    ticker: str = "",
    year: int | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "task_id": task_id,
        "percent": percent,
        "stage": stage,
        "status": status,
        "ticker": ticker,
        "year": year,
        "updated_at": time.time(),
        "error": error,
    }
    r.set(f"progress:{task_id}", json.dumps(payload), ex=KEY_TTL)


@celery_app.task(name="auditpilot.run_due_diligence", bind=True)
def run_due_diligence(self, ticker: str, year: int | None, task_id: str) -> dict[str, Any]:
    """Execute the full due-diligence pipeline for one company."""
    r = _redis()
    ticker = ticker.strip().upper()
    started = time.time()

    try:
        # --- 0%: fetch SEC filing -------------------------------------------------
        set_progress(r, task_id, 0, "Fetching SEC 10-K filing", ticker=ticker, year=year)
        chunks = edgar.fetch_10k_chunks(ticker, year)

        # --- 20%: embed into ChromaDB ---------------------------------------------
        set_progress(r, task_id, 20, "Embedding into ChromaDB", ticker=ticker, year=year)
        vector_store.ingest(chunks)

        # --- token sink for live memo streaming -----------------------------------
        r.delete(f"stream:{task_id}")

        def token_callback(token: str) -> None:
            r.append(f"stream:{task_id}", token)
            r.expire(f"stream:{task_id}", KEY_TTL)

        initial_state: dict[str, Any] = {
            "ticker": ticker,
            "filing_text_chunks": [c["text"] for c in chunks],
            "iteration": 0,
        }
        config = {
            "configurable": {"token_callback": token_callback, "thread_id": task_id},
            "run_name": f"auditpilot-{ticker}",
            "tags": ["auditpilot", ticker],
        }

        # --- 35-85%: stream the graph, updating progress per node -----------------
        set_progress(r, task_id, 35, "Running Planner", ticker=ticker, year=year)
        final_state: dict[str, Any] = dict(initial_state)
        for update in compiled_graph.stream(initial_state, config=config):
            for node_name, node_output in update.items():
                if isinstance(node_output, dict):
                    final_state.update(node_output)
                percent = NODE_PROGRESS.get(node_name)
                if percent is not None:
                    stage = {
                        "planner": "Running Planner",
                        "risk_analyst": "Running Risk Analyst",
                        "critic": "Running Critic (hallucination control)",
                        "memo_writer": "Writing memo",
                    }.get(node_name, node_name)
                    set_progress(r, task_id, percent, stage, ticker=ticker, year=year)

        memo = final_state.get("final_memo", "")

        # --- 100%: complete -------------------------------------------------------
        r.set(f"memo:{task_id}", memo, ex=KEY_TTL)
        # Ensure the live stream buffer holds the full memo for late subscribers.
        if memo and not r.exists(f"stream:{task_id}"):
            r.set(f"stream:{task_id}", memo, ex=KEY_TTL)

        elapsed = round(time.time() - started, 1)
        set_progress(
            r,
            task_id,
            100,
            f"Complete in {elapsed}s",
            status="complete",
            ticker=ticker,
            year=year,
        )
        return {
            "task_id": task_id,
            "ticker": ticker,
            "year": year,
            "elapsed_seconds": elapsed,
            "faithfulness_score": (final_state.get("critic_feedback") or {}).get(
                "faithfulness_score"
            ),
        }

    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        set_progress(
            r,
            task_id,
            0,
            "Failed",
            status="error",
            ticker=ticker,
            year=year,
            error=str(exc),
        )
        raise

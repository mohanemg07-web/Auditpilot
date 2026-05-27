"""FastAPI backend for AuditPilot.

Endpoints:
    POST /analyze            -> {"task_id": ...}  (enqueues a Celery job)
    GET  /status/{task_id}   -> progress JSON + percentage
    GET  /result/{task_id}   -> final memo markdown
    GET  /stream/{task_id}   -> Server-Sent Events streaming memo tokens live
    GET  /health             -> health check

Streaming is implemented with sse-starlette's EventSourceResponse over an async
Redis poll of the ``stream:{task_id}`` buffer the Celery worker appends to.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.config import settings
from backend.tasks import run_due_diligence

app = FastAPI(
    title="AuditPilot API",
    description="Autonomous Financial Due Diligence Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STREAM_POLL_INTERVAL = 0.3
_STREAM_MAX_SECONDS = 300


def _aredis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


# --- request/response models ----------------------------------------------
class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    year: int | None = Field(default=None, examples=[2023])


class AnalyzeResponse(BaseModel):
    task_id: str


# --- endpoints --------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "auditpilot"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    task_id = uuid.uuid4().hex
    # Seed an initial progress record so /status works immediately.
    r = _aredis()
    await r.set(
        f"progress:{task_id}",
        json.dumps(
            {
                "task_id": task_id,
                "percent": 0,
                "stage": "Queued",
                "status": "queued",
                "ticker": ticker,
                "year": req.year,
            }
        ),
        ex=60 * 60 * 24,
    )
    await r.aclose()

    run_due_diligence.apply_async(args=[ticker, req.year, task_id], task_id=task_id)
    return AnalyzeResponse(task_id=task_id)


@app.get("/status/{task_id}")
async def status(task_id: str) -> dict:
    r = _aredis()
    raw = await r.get(f"progress:{task_id}")
    await r.aclose()
    if raw is None:
        raise HTTPException(status_code=404, detail="task_id not found")
    return json.loads(raw)


@app.get("/result/{task_id}")
async def result(task_id: str) -> dict:
    r = _aredis()
    memo = await r.get(f"memo:{task_id}")
    raw_progress = await r.get(f"progress:{task_id}")
    await r.aclose()
    if memo is None:
        # Distinguish "still running" from "unknown task".
        if raw_progress is None:
            raise HTTPException(status_code=404, detail="task_id not found")
        prog = json.loads(raw_progress)
        raise HTTPException(
            status_code=425,  # Too Early
            detail=f"Memo not ready (status={prog.get('status')}, "
            f"{prog.get('percent')}%)",
        )
    return {"task_id": task_id, "memo": memo}


@app.get("/stream/{task_id}")
async def stream(task_id: str, request: Request) -> EventSourceResponse:
    """Stream memo tokens as Server-Sent Events.

    Replays whatever has already been generated, then emits deltas live until the
    task completes or errors.
    """

    async def event_generator():
        r = _aredis()
        last_len = 0
        waited = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    break

                buffer = await r.get(f"stream:{task_id}") or ""
                if len(buffer) > last_len:
                    yield {"event": "token", "data": buffer[last_len:]}
                    last_len = len(buffer)

                raw_progress = await r.get(f"progress:{task_id}")
                prog = json.loads(raw_progress) if raw_progress else {}
                state = prog.get("status")

                if state == "complete":
                    yield {"event": "done", "data": json.dumps({"percent": 100})}
                    break
                if state == "error":
                    yield {"event": "error", "data": prog.get("error", "unknown error")}
                    break

                waited += _STREAM_POLL_INTERVAL
                if waited >= _STREAM_MAX_SECONDS:
                    yield {"event": "error", "data": "stream timeout"}
                    break
                await asyncio.sleep(_STREAM_POLL_INTERVAL)
        finally:
            await r.aclose()

    return EventSourceResponse(event_generator())

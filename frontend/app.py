"""AuditPilot Streamlit dashboard.

Workflow:
  1. User enters a ticker + year and clicks Analyze -> POST /analyze.
  2. Progress bar polls GET /status every ~2s through the early pipeline stages.
  3. The memo is streamed live from GET /stream (SSE) into the page.
  4. The authoritative final memo (GET /result) is rendered as markdown.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import httpx
import streamlit as st

try:
    # When launched via `streamlit run frontend/app.py`, frontend/ is on sys.path.
    from sample_memo import SAMPLE_MEMO, SAMPLE_META
except ImportError:  # when imported as a package module
    from frontend.sample_memo import SAMPLE_MEMO, SAMPLE_META

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = 2.0

MODEL_INFO = {
    "Planner": "gpt-4o",
    "Risk Analyst": "gpt-4o",
    "Critic": "gpt-4o-mini",
    "Memo Writer": "gpt-4o",
    "Embeddings": "text-embedding-3-small",
}

st.set_page_config(page_title="AuditPilot", page_icon="📊", layout="wide")


# --- helpers ---------------------------------------------------------------
def post_analyze(ticker: str, year: int) -> str:
    resp = httpx.post(
        f"{BACKEND_URL}/analyze",
        json={"ticker": ticker, "year": year},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["task_id"]


def get_status(task_id: str) -> dict:
    resp = httpx.get(f"{BACKEND_URL}/status/{task_id}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_result(task_id: str) -> str:
    resp = httpx.get(f"{BACKEND_URL}/result/{task_id}", timeout=30)
    resp.raise_for_status()
    return resp.json().get("memo", "")


def iter_sse(task_id: str):
    """Yield (event, data) tuples from the SSE stream, parsing per the SSE spec."""
    url = f"{BACKEND_URL}/stream/{task_id}"
    with httpx.stream("GET", url, timeout=None) as resp:
        event = "message"
        data_lines: list[str] = []
        for line in resp.iter_lines():
            if line == "":
                if data_lines:
                    yield event, "\n".join(data_lines)
                event, data_lines = "message", []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[len("event:") :].lstrip()
            elif line.startswith("data:"):
                payload = line[len("data:") :]
                if payload.startswith(" "):  # SSE strips exactly one leading space
                    payload = payload[1:]
                data_lines.append(payload)


# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Run details")
    st.caption(f"Backend: `{BACKEND_URL}`")
    demo_mode = st.toggle(
        "🎬 Demo Mode (no API)",
        value=False,
        help="Render a packaged AAPL sample memo without calling the backend, "
        "OpenAI, or SEC. Useful for showing the UI end-to-end with no keys.",
    )
    sidebar_task = st.empty()
    sidebar_time = st.empty()
    st.divider()
    st.subheader("Model configuration")
    for role, model in MODEL_INFO.items():
        st.text(f"{role}: {model}")
    st.divider()
    st.caption("Orchestration: LangGraph 5-node pipeline")
    st.caption("Tracing: LangSmith")


# --- main ------------------------------------------------------------------
st.title("AuditPilot — Financial Due Diligence Agent")
st.write(
    "Autonomous SEC 10-K analysis: fetch → embed → plan → analyze risk → "
    "critique → write memo."
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
with col2:
    year = st.selectbox("Year", options=[2024, 2023, 2022, 2021, 2020], index=1)
with col3:
    st.write("")
    st.write("")
    analyze = st.button("Analyze", type="primary", use_container_width=True)

progress_bar = st.progress(0, text="Idle")
status_msg = st.empty()
st.divider()
st.subheader("Due Diligence Memo")
memo_box = st.empty()

if analyze and ticker and demo_mode:
    # ---- Demo Mode: simulate the pipeline + render the packaged fixture ----
    sidebar_task.info(f"Task ID:\n`{SAMPLE_META['task_id']}`")
    sidebar_time.caption(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for pct, stage in [
        (0, "Fetching SEC 10-K filing"),
        (20, "Embedding into ChromaDB"),
        (35, "Running Planner"),
        (55, "Running Risk Analyst"),
        (70, "Running Critic (hallucination control)"),
        (85, "Writing memo"),
    ]:
        progress_bar.progress(pct, text=f"{pct}% — {stage}")
        status_msg.info(f"{stage}…")
        time.sleep(0.35)

    shown = ""
    for para in SAMPLE_MEMO.split("\n\n"):
        shown += para + "\n\n"
        memo_box.markdown(shown)
        time.sleep(0.05)
    memo_box.markdown(SAMPLE_MEMO)
    progress_bar.progress(100, text="100% — Complete (demo)")
    status_msg.success(
        f"Demo complete · faithfulness {SAMPLE_META['faithfulness_score']} · "
        f"{SAMPLE_META['elapsed_seconds']}s (simulated)"
    )
    st.stop()

if analyze and ticker:
    try:
        task_id = post_analyze(ticker, int(year))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to start analysis: {exc}")
        st.stop()

    sidebar_task.info(f"Task ID:\n`{task_id}`")
    sidebar_time.caption(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase A: poll progress through the early stages.
    terminal = False
    while True:
        try:
            prog = get_status(task_id)
        except Exception as exc:  # noqa: BLE001
            status_msg.warning(f"Waiting for task… ({exc})")
            time.sleep(POLL_INTERVAL)
            continue

        percent = int(prog.get("percent", 0))
        stage = prog.get("stage", "")
        state = prog.get("status", "running")
        progress_bar.progress(min(percent, 100), text=f"{percent}% — {stage}")

        if state == "error":
            status_msg.error(f"Pipeline failed: {prog.get('error')}")
            terminal = True
            break
        if state == "complete" or percent >= 85:
            status_msg.info(f"{stage}")
            break
        status_msg.info(f"{stage}…")
        time.sleep(POLL_INTERVAL)

    # Phase B: live-stream the memo tokens.
    if not terminal:
        memo_text = ""
        try:
            for event, data in iter_sse(task_id):
                if event == "token":
                    memo_text += data
                    memo_box.markdown(memo_text)
                elif event == "error":
                    status_msg.error(f"Stream error: {data}")
                    break
                elif event == "done":
                    break
        except Exception as exc:  # noqa: BLE001
            status_msg.warning(f"Stream interrupted, fetching final result… ({exc})")

        # Phase C: render the authoritative final memo.
        try:
            final = get_result(task_id)
            if final:
                memo_box.markdown(final)
            progress_bar.progress(100, text="100% — Complete")
            status_msg.success("Analysis complete.")
        except Exception as exc:  # noqa: BLE001
            if memo_text:
                status_msg.success("Analysis complete (streamed).")
            else:
                status_msg.error(f"Could not fetch final memo: {exc}")
elif analyze and not ticker:
    st.warning("Please enter a ticker symbol.")

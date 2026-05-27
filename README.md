# AuditPilot — Autonomous Financial Due Diligence Agent

AuditPilot is a 5-node [LangGraph](https://langchain-ai.github.io/langgraph/)
agentic pipeline that performs automated due diligence on a public company:

1. **Fetch** the company's most recent SEC **10-K** filing from EDGAR.
2. **Embed** the filing text into **ChromaDB** with OpenAI `text-embedding-3-small`.
3. **Reason** through a multi-agent graph: **Planner → Risk Analyst → Critic → Memo Writer**
   (the Critic can loop back to the Risk Analyst to correct hallucinations).
4. **Stream** a structured markdown due-diligence memo live to a **Streamlit** dashboard.

```
            ┌──────────┐     ┌──────────────┐     ┌────────┐
 START ───▶ │ Planner  │ ──▶ │ Risk Analyst │ ──▶ │ Critic │
            │  GPT-4o  │     │    GPT-4o    │     │ 4o-mini│
            └──────────┘     └──────────────┘     └───┬────┘
                                   ▲                   │ hallucination & iter<2
                                   └───────────────────┤
                                                       │ else
                                                 ┌─────▼──────┐
                                                 │ Memo Writer│ ──▶ END
                                                 │   GPT-4o   │
                                                 └────────────┘
```

## Architecture

| Layer        | Tech                                               |
|--------------|----------------------------------------------------|
| Orchestration| LangGraph `StateGraph` (5 nodes, conditional loop) |
| LLMs         | GPT-4o (planner/analyst/memo), GPT-4o-mini (critic)|
| Embeddings   | OpenAI `text-embedding-3-small`                    |
| Vector store | ChromaDB (persistent, `sec_filings` collection)    |
| Data source  | SEC EDGAR submissions API (ticker → CIK → 10-K)    |
| API          | FastAPI (+ SSE streaming via `sse-starlette`)      |
| Task queue   | Celery + Redis (broker & result backend)           |
| Frontend     | Streamlit live dashboard                           |
| Tracing      | LangSmith (auto via `LANGCHAIN_*` env vars)        |
| Eval         | RAGAS faithfulness + answer relevancy              |

### Project layout
```
backend/
  agents/   planner.py · risk_analyst.py · critic.py · memo_writer.py · graph.py
  tools/    edgar.py · vector_store.py
  config.py · tasks.py · main.py
frontend/app.py
evals/ragas_eval.py
Dockerfile.backend · Dockerfile.frontend · docker-compose.yml · render.yaml
requirements.txt · .env.example
```

> **EDGAR note:** rather than the fragile `efts.sec.gov` full-text endpoint, the
> fetcher resolves *ticker → CIK* via `company_tickers.json`, then uses the
> `data.sec.gov` submissions API to locate the latest (or per-year) 10-K and
> download its primary HTML document. Same chunk/metadata output contract.

## Quick start (Docker)

```bash
cp .env.example .env          # then edit .env with your keys
docker-compose up --build     # starts redis + backend + worker + frontend
```

- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs

The four services are `redis`, `backend` (FastAPI :8000), `worker` (Celery), and
`frontend` (Streamlit :8501); they share the `.env` file.

### Required environment variables
See [.env.example](.env.example). At minimum set:

- `OPENAI_API_KEY` — for LLM + embedding calls
- `SEC_USER_AGENT` — **required by the SEC**, e.g. `Jane Doe jane@example.com`
- `LANGCHAIN_API_KEY` — optional, enables LangSmith tracing

## API

| Method | Path                | Description                                   |
|--------|---------------------|-----------------------------------------------|
| POST   | `/analyze`          | `{"ticker":"AAPL","year":2023}` → `{task_id}` |
| GET    | `/status/{task_id}` | progress `{percent, stage, status, …}`        |
| GET    | `/result/{task_id}` | final memo markdown                           |
| GET    | `/stream/{task_id}` | SSE stream of memo tokens                     |
| GET    | `/health`           | health check                                  |

```bash
# enqueue
curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
     -d '{"ticker":"AAPL","year":2023}'
# poll
curl localhost:8000/status/<task_id>
```

Pipeline progress stages: `0% fetch → 20% embed → 35% plan → 55% risk →
70% critic → 85% memo → 100% complete` (target: < 85 s end-to-end).

## Local development (without Docker)

Requires **Python 3.11+** and a running Redis.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # set REDIS_URL=redis://localhost:6379/0

# 3 terminals:
celery -A backend.tasks.celery_app worker --loglevel=info   # worker
uvicorn backend.main:app --reload                            # API :8000
streamlit run frontend/app.py                                # UI :8501
```

## Evaluation (RAGAS)

```bash
python -m evals.ragas_eval --limit 3      # quick run
python -m evals.ragas_eval                # full 30-company sample
```
Writes `evals/results.csv` with per-row `faithfulness` / `answer_relevancy` and
reports whether mean faithfulness clears the **0.91** target.

## Deployment (Render)

[`render.yaml`](render.yaml) is a Blueprint defining the backend (web), Celery
worker (background, with the persistent Chroma disk), a managed Redis instance,
and the Streamlit frontend (web). Push to GitHub, create a Render **Blueprint**,
and fill the `sync:false` secrets (`OPENAI_API_KEY`, `LANGCHAIN_API_KEY`,
`SEC_USER_AGENT`, and the frontend's `BACKEND_URL`).

## Memo structure

Every memo contains: **Executive Summary**, **Company Overview**, **Risk Analysis**
(Market, Credit, Liquidity, Operational, Regulatory/Legal, Strategic), **Key
Findings**, **Recommendation**, and **Appendix: Data Sources** — with figures
cited from the filing and verified by the Critic node.

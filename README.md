<div align="center">

# 🛰️ AuditPilot

### Autonomous Financial Due Diligence Agent

*Point it at a ticker. It pulls the 10-K, reasons through six risk categories with a self-correcting multi-agent graph, and streams back a cited due-diligence memo.*

[![CI](https://github.com/mohanemg07-web/Auditpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mohanemg07-web/Auditpilot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GPT-4o](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)](https://platform.openai.com/)

</div>

---

AuditPilot is a 5-node [LangGraph](https://langchain-ai.github.io/langgraph/)
agentic pipeline that performs automated due diligence on a public company:

1. **Fetch** the company's most recent SEC **10-K** filing from EDGAR.
2. **Embed** the filing text into **ChromaDB** with OpenAI `text-embedding-3-small`.
3. **Reason** through a multi-agent graph: **Planner → Risk Analyst → Critic → Memo Writer**
   (the Critic can loop back to the Risk Analyst to correct hallucinations).
4. **Stream** a structured markdown due-diligence memo live to a **Streamlit** dashboard.

## Architecture

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

```
                 ┌───────────┐        ┌───────────────────────────┐
   Browser ────▶ │ Streamlit │ ─REST─▶│ FastAPI  (POST /analyze)  │
   (SSE memo)    │   :8501   │ ◀─SSE──│ + SSE     :8000           │
                 └───────────┘        └────────────┬──────────────┘
                                                    │ enqueue task
                                              ┌─────▼─────┐
                                              │  Redis    │ broker + memo/progress buffers
                                              └─────┬─────┘
                                                    │ consume
                                       ┌────────────▼─────────────┐
                                       │ Celery worker            │
                                       │  ↳ LangGraph pipeline     │
                                       │  ↳ EDGAR fetch · ChromaDB │
                                       └───────────────────────────┘
```

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

> **EDGAR note:** rather than the fragile `efts.sec.gov` full-text endpoint, the
> fetcher resolves *ticker → CIK* via `company_tickers.json`, then uses the
> `data.sec.gov` submissions API to locate the latest (or per-year) 10-K and
> download its primary HTML document.

## Quick start (Docker)

```bash
git clone https://github.com/mohanemg07-web/Auditpilot.git
cd Auditpilot
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
| GET    | `/health`           | health check                                  |
| POST   | `/analyze`          | `{"ticker":"AAPL","year":2023}` → `{task_id}` |
| GET    | `/status/{task_id}` | progress `{percent, stage, status, …}`        |
| GET    | `/result/{task_id}` | final memo markdown                           |
| GET    | `/stream/{task_id}` | SSE stream of memo tokens                     |

```bash
# enqueue
curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
     -d '{"ticker":"AAPL","year":2023}'
# poll
curl localhost:8000/status/<task_id>
```

Pipeline progress stages: `0% fetch → 20% embed → 35% plan → 55% risk →
70% critic → 85% memo → 100% complete` (target: < 85 s end-to-end).

## Viewing LangSmith traces

Every LangGraph run is traced automatically — no per-call wiring. On import,
[`backend/config.py`](backend/config.py) exports the `LANGCHAIN_*` variables into
the process environment so LangChain/LangGraph emit traces to LangSmith.

1. Set `LANGCHAIN_API_KEY` in your `.env` (get a key at
   [smith.langchain.com](https://smith.langchain.com) → *Settings → API Keys*).
2. Keep `LANGCHAIN_TRACING_V2=true` (default) and `LANGCHAIN_PROJECT=auditpilot`.
3. Run an analysis, then open
   [smith.langchain.com](https://smith.langchain.com) → project **`auditpilot`**.

Each trace shows the full node tree (Planner → Risk Analyst → Critic →
Memo Writer), every prompt/completion, token counts, latency per node, and any
Critic → Risk Analyst correction loops.

## 💰 Estimated cost per run

A typical single-company run costs **~$0.19** in OpenAI API spend:

| Item                                  | Model            | Approx. cost |
|---------------------------------------|------------------|--------------|
| Embedding the 10-K filing             | `text-embedding-3-small` | ~$0.01 |
| Planner                               | `gpt-4o`         | ~$0.02 |
| Risk Analyst (6 categories, ×RAG)     | `gpt-4o`         | ~$0.09 |
| Critic (+ up to 1 loop)               | `gpt-4o-mini`    | ~$0.01 |
| Memo Writer                           | `gpt-4o`         | ~$0.06 |
| **Total**                             |                  | **≈ $0.19** |

> Cost varies with filing size and whether the Critic triggers a correction loop
> (capped at `MAX_CRITIC_ITERATIONS=2`). Swap any node's model via the
> `*_MODEL` env vars in [.env.example](.env.example) to trade cost for quality.

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

## Project structure

```
AuditPilot/
├── backend/
│   ├── __init__.py
│   ├── config.py              # typed Settings, exports LANGCHAIN_* env, 6 risk categories
│   ├── main.py                # FastAPI app: /analyze /status /result /stream /health
│   ├── tasks.py               # Celery task: runs the LangGraph pipeline, writes progress
│   ├── agents/
│   │   ├── __init__.py        # shared LLM factory + JSON-mode invoke helpers
│   │   ├── planner.py         # Planner node (GPT-4o) — builds the analysis plan
│   │   ├── risk_analyst.py    # Risk Analyst node (GPT-4o) — 6 categories, RAG-grounded
│   │   ├── critic.py          # Critic node (GPT-4o-mini) + route_after_critic loop logic
│   │   ├── memo_writer.py     # Memo Writer node (GPT-4o) — streams final markdown memo
│   │   └── graph.py           # StateGraph wiring + compiled_graph singleton
│   └── tools/
│       ├── __init__.py
│       ├── edgar.py           # SEC EDGAR fetch: ticker → CIK → latest 10-K → chunks
│       └── vector_store.py    # ChromaDB persistent store: embed / retrieve
├── frontend/
│   ├── app.py                 # Streamlit dashboard (SSE live memo)
│   └── sample_memo.py         # bundled demo memo for offline preview
├── evals/
│   ├── __init__.py
│   └── ragas_eval.py          # RAGAS faithfulness + answer-relevancy harness
├── .github/
│   └── workflows/
│       └── ci.yml             # compile + docker-compose config check on push to main
├── Dockerfile.backend         # image for backend + worker (shared)
├── Dockerfile.frontend        # image for the Streamlit UI
├── docker-compose.yml         # redis + backend + worker + frontend
├── render.yaml                # Render Blueprint (deploy)
├── requirements.txt
├── .env.example               # config template (copy to .env)
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to
run the stack locally, run the RAGAS evals, follow the code style, and add a new
risk category.

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

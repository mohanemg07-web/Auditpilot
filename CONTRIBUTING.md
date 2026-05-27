# Contributing to AuditPilot

Thanks for your interest in improving AuditPilot! This guide covers running the
project locally, running evals, the code style we follow, and the most common
extension task — adding a new risk category.

## Running locally

### Option A — Docker (recommended)

```bash
git clone https://github.com/mohanemg07-web/Auditpilot.git
cd Auditpilot
cp .env.example .env          # fill in OPENAI_API_KEY + SEC_USER_AGENT
docker-compose up --build
```

- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs

### Option B — bare metal (Python 3.11+ and Redis)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # REDIS_URL=redis://localhost:6379/0

# run each in its own terminal:
celery -A backend.tasks.celery_app worker --loglevel=info
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

Required env vars (see [.env.example](.env.example)):

| Var                | Purpose                                              |
|--------------------|------------------------------------------------------|
| `OPENAI_API_KEY`   | LLM + embedding calls (required)                     |
| `SEC_USER_AGENT`   | Required by the SEC, e.g. `Jane Doe jane@example.com`|
| `LANGCHAIN_API_KEY`| Optional — enables LangSmith tracing                 |

## Running the evals

AuditPilot uses [RAGAS](https://docs.ragas.io/) to measure how faithful the
generated memos are to the source filing.

```bash
python -m evals.ragas_eval --limit 3      # quick smoke run (3 companies)
python -m evals.ragas_eval                # full ~30-company sample
```

This writes `evals/results.csv` with per-row `faithfulness` and
`answer_relevancy`, and reports whether mean faithfulness clears the **0.91**
target. Please run at least the `--limit 3` evals before opening a PR that
touches any agent prompt, the retrieval logic, or the graph wiring.

## Code style

- **Python 3.11+**, `from __future__ import annotations` at the top of modules.
- **Type hints** on all public functions; prefer modern syntax (`list[str]`,
  `dict[str, Any]`, `X | None`).
- **Docstrings**: module-level docstring describing the node/tool's role, plus
  docstrings on public functions. Match the existing voice — concise, factual.
- **Config over constants**: tunable values live in
  [`backend/config.py`](backend/config.py) (the `Settings` object), driven by env
  vars — don't hardcode models, top-k, or thresholds inside nodes.
- **LLM access** goes through the shared `make_llm` / `invoke_json` helpers in
  [`backend/agents/__init__.py`](backend/agents/__init__.py); don't instantiate
  clients ad hoc.
- **Grounding**: agent prompts must require claims to be grounded in retrieved
  source excerpts. Never relax the "do not invent figures" rule.
- Keep diffs focused; the CI workflow byte-compiles `backend/`, `frontend/`, and
  `evals/` and validates `docker-compose.yml` on every push and PR to `main`.

## Adding a new risk category

Risk categories are defined in one source of truth and referenced in a few
places. To add one (e.g. `esg_risk` / "ESG Risk"):

1. **`backend/config.py`** — add the `key: "Label"` entry to the
   `risk_categories` property. This is the canonical list the Risk Analyst and
   Planner iterate over.

2. **`backend/agents/risk_analyst.py`** — add a matching entry to
   `_CATEGORY_QUERIES` using the **same key**. This is the RAG retrieval query
   that seeds ChromaDB search for the new category. *(A missing key here raises
   a `KeyError`, so this step is required.)*

3. **`backend/agents/planner.py`** — the planner prompt enumerates the categories
   in prose (around line 24); update that list so the plan accounts for the new
   category.

4. **`backend/agents/memo_writer.py`** — the memo template includes a section
   heading per category (e.g. `### Market Risk`); add the new heading so it
   appears in the final memo.

5. **Verify** with `python -m evals.ragas_eval --limit 3` and a manual run, then
   update the "Memo structure" list in [README.md](README.md).

## Pull requests

1. Fork and create a branch off `main`.
2. Make your change; keep it focused and add/update docstrings.
3. Run `python -m compileall backend/ frontend/ evals/` and, if you touched
   agents/retrieval/graph, the `--limit 3` evals.
4. Open a PR describing **what** changed and **why**. CI must be green.

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).

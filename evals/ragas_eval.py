"""RAGAS faithfulness / answer-relevancy evaluation for AuditPilot.

For a sample of companies (default 30), this script builds a RAGAS evaluation
dataset by, for each company and each of the six risk categories:

    question -> the category's retrieval query
    contexts -> the top-k chunks ChromaDB returns for that query/ticker
    answer   -> a grounded RAG answer the risk model generates from those contexts

It then scores ``faithfulness`` and ``answer_relevancy`` with the RAGAS library,
writes a per-row + aggregate CSV to ``evals/results.csv``, and reports whether the
mean faithfulness clears the 0.91 target.

Targets the RAGAS 0.1.x dataset schema (question/answer/contexts/ground_truth).

Usage:
    python -m evals.ragas_eval                 # full 30-company sample
    python -m evals.ragas_eval --limit 3       # quick/cheap run
    python -m evals.ragas_eval --tickers AAPL MSFT --year 2023
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents import make_llm
from backend.agents.risk_analyst import _CATEGORY_QUERIES
from backend.config import settings
from backend.tools import edgar, vector_store

# A representative 30-company sample across sectors.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
    "GS", "XOM", "CVX", "JNJ", "PFE", "MRK", "UNH", "WMT", "COST", "HD",
    "PG", "KO", "PEP", "DIS", "NFLX", "INTC", "AMD", "CSCO", "ORCL", "IBM",
]

FAITHFULNESS_TARGET = 0.91
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

ANSWER_SYSTEM = (
    "You are a financial analyst. Answer the question about the company's risk "
    "profile using ONLY the provided source excerpts from its SEC 10-K. Cite "
    "specific figures where present. If the sources do not address it, say so."
)


def _generate_answer(question: str, contexts: list[str]) -> str:
    llm = make_llm(settings.risk_model, temperature=0.0)
    source = "\n\n---\n\n".join(contexts)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nSource excerpts:\n{source}"),
    ]
    resp = llm.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)


def build_samples(tickers: list[str], year: int | None) -> list[dict]:
    """Ingest each filing if needed, then build RAGAS rows per risk category."""
    rows: list[dict] = []
    for ticker in tickers:
        ticker = ticker.upper()
        try:
            if not vector_store.has_ticker(ticker):
                print(f"[{ticker}] fetching + embedding 10-K…", flush=True)
                chunks = edgar.fetch_10k_chunks(ticker, year)
                vector_store.ingest(chunks)
            else:
                print(f"[{ticker}] already ingested, reusing.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{ticker}] SKIP — fetch/ingest failed: {exc}", flush=True)
            continue

        for cat_key, query in _CATEGORY_QUERIES.items():
            contexts = [
                c["text"]
                for c in vector_store.retrieve(query, ticker=ticker, n_results=settings.retrieval_top_k)
            ]
            if not contexts:
                continue
            question = f"What are the {cat_key.replace('_', ' ')} disclosures for {ticker}?"
            answer = _generate_answer(query, contexts)
            rows.append(
                {
                    "ticker": ticker,
                    "category": cat_key,
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": "",  # not required by faithfulness/answer_relevancy
                }
            )
    return rows


def run_ragas(rows: list[dict]) -> pd.DataFrame:
    """Score the dataset with RAGAS and return a per-row DataFrame."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    # RAGAS returns an object convertible to a pandas DataFrame.
    scores_df = result.to_pandas()
    return scores_df


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS eval for AuditPilot")
    parser.add_argument("--tickers", nargs="*", default=None, help="override ticker list")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--limit", type=int, default=None, help="cap number of tickers")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    if args.limit:
        tickers = tickers[: args.limit]

    print(f"Evaluating {len(tickers)} companies (year={args.year})…")
    rows = build_samples(tickers, args.year)
    if not rows:
        print("No samples could be built (check OPENAI_API_KEY / SEC access).")
        return 1

    scores_df = run_ragas(rows)
    scores_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(scores_df)} rows to {OUTPUT_CSV}")

    mean_faith = float(scores_df["faithfulness"].mean())
    mean_rel = (
        float(scores_df["answer_relevancy"].mean())
        if "answer_relevancy" in scores_df
        else float("nan")
    )
    print(f"Mean faithfulness     : {mean_faith:.4f} (target >= {FAITHFULNESS_TARGET})")
    print(f"Mean answer_relevancy : {mean_rel:.4f}")
    status = "PASS" if mean_faith >= FAITHFULNESS_TARGET else "BELOW TARGET"
    print(f"Result: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

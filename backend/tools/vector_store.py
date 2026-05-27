"""ChromaDB-backed vector store for SEC filing chunks.

Persistent local store (``CHROMA_PERSIST_DIR``), single collection
``sec_filings``. Embeddings are produced with OpenAI ``text-embedding-3-small``
via ``langchain-openai`` (batched, with exponential-backoff retries on rate
limits). Chunks are upserted with deterministic IDs ``f"{ticker}_{chunk_index}"``
so re-ingesting the same filing is idempotent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings


@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.openai_api_key or None,
    )


@lru_cache(maxsize=1)
def _client() -> chromadb.api.client.Client:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


@lru_cache(maxsize=1)
def _collection():
    return _client().get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _embed_documents(texts: list[str]) -> list[list[float]]:
    return _embeddings().embed_documents(texts)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _embed_query(text: str) -> list[float]:
    return _embeddings().embed_query(text)


def ingest(chunks: list[dict[str, Any]], batch_size: int = 100) -> int:
    """Embed and upsert chunk dicts into the ``sec_filings`` collection.

    Each chunk dict must contain: ticker, chunk_index, text, filing_date, section.
    Returns the number of chunks ingested.
    """
    if not chunks:
        return 0

    collection = _collection()
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"{c['ticker']}_{c['chunk_index']}" for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "ticker": c["ticker"],
                "filing_date": c.get("filing_date", ""),
                "section": c.get("section", "General"),
                "chunk_index": int(c["chunk_index"]),
            }
            for c in batch
        ]
        embeddings = _embed_documents(documents)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total += len(batch)
    return total


def retrieve(query: str, ticker: str, n_results: int = None) -> list[dict[str, Any]]:
    """Return the top-``n_results`` chunks for ``query`` scoped to ``ticker``.

    Output items: {"text", "section", "filing_date", "chunk_index", "score"}.
    """
    n_results = n_results or settings.retrieval_top_k
    collection = _collection()

    query_embedding = _embed_query(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"ticker": ticker.upper()},
        include=["documents", "metadatas", "distances"],
    )

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "section": (meta or {}).get("section", "General"),
                "filing_date": (meta or {}).get("filing_date", ""),
                "chunk_index": (meta or {}).get("chunk_index", -1),
                # cosine distance -> rough similarity score
                "score": round(1.0 - float(dist), 4),
            }
        )
    return out


def has_ticker(ticker: str) -> bool:
    """True if any chunks for ``ticker`` are already stored."""
    try:
        res = _collection().get(where={"ticker": ticker.upper()}, limit=1)
        return bool(res.get("ids"))
    except Exception:
        return False

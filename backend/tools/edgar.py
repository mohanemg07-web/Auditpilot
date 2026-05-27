"""SEC EDGAR 10-K fetcher.

Implements the *reliable* canonical path for retrieving a company's most recent
10-K filing (the literal ``efts.sec.gov`` full-text endpoint in the original spec
is fragile and frequently returns nothing fetchable):

    ticker  --company_tickers.json-->  CIK
    CIK     --data.sec.gov/submissions-->  list of filings (pick latest 10-K for year)
    filing  --www.sec.gov/Archives-->  primary HTML document
    HTML    --BeautifulSoup-->  clean text  --tiktoken-->  800-token chunks (100 overlap)

Every request sends the SEC-mandated ``User-Agent`` header and is wrapped in
exponential-backoff retries. The output contract is a list of chunk dicts:

    {"ticker", "filing_date", "section", "chunk_index", "text"}
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests
import tiktoken
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings

# --- SEC endpoints ---------------------------------------------------------
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# SEC asks clients to stay <= 10 requests/second. We pace conservatively.
_MIN_REQUEST_INTERVAL = 0.15
_last_request_ts = 0.0

# Cache the (large) ticker->CIK map for the lifetime of the process.
_ticker_cik_cache: dict[str, dict[str, Any]] | None = None


def _throttle() -> None:
    """Space out requests so we stay polite to SEC servers."""
    global _last_request_ts
    delta = time.time() - _last_request_ts
    if delta < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - delta)
    _last_request_ts = time.time()


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _get(url: str, *, host: str) -> requests.Response:
    """GET with SEC headers, throttling and exponential-backoff retries."""
    _throttle()
    headers = {
        "User-Agent": settings.sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


# --- ticker -> CIK ----------------------------------------------------------
def _load_ticker_map() -> dict[str, dict[str, Any]]:
    global _ticker_cik_cache
    if _ticker_cik_cache is None:
        resp = _get(COMPANY_TICKERS_URL, host="www.sec.gov")
        raw = resp.json()
        # company_tickers.json is keyed by arbitrary index ints.
        _ticker_cik_cache = {
            row["ticker"].upper(): row for row in raw.values() if "ticker" in row
        }
    return _ticker_cik_cache


def get_cik(ticker: str) -> int:
    """Resolve a stock ticker to its zero-padded SEC CIK integer."""
    ticker = ticker.strip().upper()
    mapping = _load_ticker_map()
    if ticker not in mapping:
        raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers map.")
    return int(mapping[ticker]["cik_str"])


# --- locate the 10-K --------------------------------------------------------
def find_10k(cik: int, year: int | None = None) -> dict[str, str]:
    """Return metadata for the target 10-K filing.

    Picks the 10-K whose filing date falls in ``year``; if none match (or year is
    None) returns the most recent 10-K available.
    """
    resp = _get(SUBMISSIONS_URL.format(cik=cik), host="data.sec.gov")
    recent = resp.json().get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    candidates: list[dict[str, str]] = []
    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        candidates.append(
            {
                "accession": accessions[i].replace("-", ""),
                "accession_dashed": accessions[i],
                "primary_document": primary_docs[i],
                "filing_date": filing_dates[i],
                "report_date": report_dates[i] if i < len(report_dates) else "",
            }
        )

    if not candidates:
        raise ValueError(f"No 10-K filings found for CIK {cik}.")

    if year is not None:
        for c in candidates:
            if c["filing_date"].startswith(str(year)) or c["report_date"].startswith(
                str(year)
            ):
                return c
    # Fallback: most recent (filings come newest-first).
    return candidates[0]


# --- fetch + clean ----------------------------------------------------------
def fetch_filing_html(cik: int, filing: dict[str, str]) -> str:
    url = ARCHIVES_BASE.format(
        cik=cik,
        accession=filing["accession"],
        document=filing["primary_document"],
    )
    resp = _get(url, host="www.sec.gov")
    return resp.text


def _clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runaway whitespace produced by tables/formatting.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --- section detection (best-effort) ---------------------------------------
_ITEM_RE = re.compile(r"\bitem\s+(\d+[a-z]?)\b", re.IGNORECASE)


def _guess_section(text: str) -> str:
    """Best-effort label of which 10-K Item a chunk belongs to."""
    match = _ITEM_RE.search(text)
    if match:
        return f"Item {match.group(1).upper()}"
    return "General"


# --- chunking ---------------------------------------------------------------
def chunk_text(
    text: str,
    ticker: str,
    filing_date: str,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Split cleaned text into overlapping token chunks with metadata."""
    chunk_tokens = chunk_tokens or settings.chunk_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    step = max(1, chunk_tokens - overlap_tokens)
    while start < len(tokens):
        window = tokens[start : start + chunk_tokens]
        chunk_str = encoding.decode(window).strip()
        if chunk_str:
            chunks.append(
                {
                    "ticker": ticker.upper(),
                    "filing_date": filing_date,
                    "section": _guess_section(chunk_str),
                    "chunk_index": index,
                    "text": chunk_str,
                }
            )
            index += 1
        start += step
    return chunks


# --- public entrypoint ------------------------------------------------------
def fetch_10k_chunks(ticker: str, year: int | None = None) -> list[dict[str, Any]]:
    """End-to-end: ticker -> latest/year 10-K -> list of chunk dicts.

    This is the single function the Celery task calls during the "fetch" stage.
    """
    cik = get_cik(ticker)
    filing = find_10k(cik, year)
    html = fetch_filing_html(cik, filing)
    text = _clean_html_to_text(html)
    if not text:
        raise ValueError(
            f"Fetched 10-K for {ticker} ({filing['filing_date']}) contained no text."
        )
    return chunk_text(text, ticker=ticker, filing_date=filing["filing_date"])


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    tkr = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    yr = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out = fetch_10k_chunks(tkr, yr)
    print(f"{tkr}: {len(out)} chunks, first section={out[0]['section']!r}")
    print(out[0]["text"][:400])

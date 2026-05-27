"""Static demo fixture for AuditPilot's Streamlit dashboard.

`SAMPLE_MEMO` is an illustrative AAPL due-diligence memo (all six risk
categories + the full required section structure) used by the dashboard's
"Demo Mode" so the UI can be shown end-to-end without an OpenAI key, a running
backend, or live SEC calls. Figures are representative and clearly labelled as a
sample in the appendix — this is NOT the output of a live pipeline run.
"""

from __future__ import annotations

SAMPLE_META = {
    "ticker": "AAPL",
    "year": 2023,
    "task_id": "demo-aapl-2023",
    "faithfulness_score": 0.94,
    "elapsed_seconds": 71.4,
}

SAMPLE_MEMO = """\
# Due Diligence Memo — Apple Inc. (AAPL)

*Source filing: Form 10-K, fiscal year ended September 30, 2023.*

## Executive Summary

Apple Inc. ("Apple", "the Company") remains one of the most financially robust
issuers in the technology sector. For fiscal 2023 the Company reported total net
sales of **$383.3 billion** (down ~2.8% from $394.3 billion in FY2022) and net
income of **$97.0 billion**, with a gross margin of **44.1%**. The balance sheet
carries **$162.1 billion** in cash, cash equivalents and marketable securities
against **$111.1 billion** of total term debt, and the Company returned over
**$99 billion** to shareholders via buybacks and dividends during the year.

Overall risk posture is assessed as **Low-to-Moderate**. The dominant risks are
strategic and operational — concentration in iPhone revenue (~52% of net sales)
and a manufacturing/supply chain heavily dependent on partners in China — rather
than financial-solvency risks. The Critic node verified all cited figures against
the source filing with a faithfulness score of **0.94**.

## Company Overview

Apple designs, manufactures and markets smartphones, personal computers, tablets,
wearables and accessories, and sells a range of related services. Reportable
segments are geographic: Americas, Europe, Greater China, Japan, and Rest of Asia
Pacific. Products accounted for **$298.1 billion** and Services for **$85.2
billion** of FY2023 net sales, with Services growing ~9% year over year and
carrying a markedly higher gross margin (**70.8%**) than Products (**36.5%**).
The active installed base surpassed **2 billion** devices, underpinning the
recurring Services franchise.

## Risk Analysis

### Market Risk
Apple is exposed to **interest-rate risk** on its large marketable-securities
portfolio and **foreign-exchange risk**, with the majority of net sales generated
outside the U.S. The 10-K quantifies a hypothetical **100 bps** adverse interest
rate move and notes the use of foreign-currency and interest-rate derivatives to
hedge exposures. A strengthening U.S. dollar pressured reported FY2023 revenue.
*Severity: Moderate · Confidence: High.*

### Credit Risk
Counterparty credit exposure arises from cash investments, derivatives and trade
receivables. The Company reports that **no single customer represented 10% or more**
of total trade receivables and that it invests in high-quality, investment-grade
instruments with concentration limits. Vendor non-trade receivables are
concentrated among a small number of partners, a monitored exposure.
*Severity: Low · Confidence: High.*

### Liquidity Risk
Liquidity is a clear strength. The Company held **$162.1 billion** in cash and
marketable securities and generated **$110.5 billion** of operating cash flow in
FY2023. Term debt of **$111.1 billion** is laddered across maturities, and the
Company maintains commercial-paper and credit-facility capacity. Debt maturities
within 12 months are comfortably covered by operating cash flow.
*Severity: Low · Confidence: High.*

### Operational Risk
The most material operational risk is **supply-chain and manufacturing
concentration**: substantially all of Apple's hardware is assembled by outsourcing
partners, primarily located in China, exposing the Company to geopolitical,
logistics, and single-region disruption risk. Additional disclosures cover
component sole-sourcing, cybersecurity/data-security threats, and dependence on
carrier and channel relationships.
*Severity: High · Confidence: High.*

### Regulatory/Legal Risk
Apple faces escalating **antitrust and platform-regulation** scrutiny globally —
including the EU Digital Markets Act, App Store conduct litigation, and ongoing
government investigations — which could compel changes to App Store economics
(its ~30% commission) and third-party payment/sideloading rules. The filing also
discloses privacy/data-protection regimes (GDPR-style) and export-control exposure.
*Severity: High · Confidence: Medium.*

### Strategic Risk
Revenue concentration is the headline strategic risk: **iPhone represented ~52%**
of net sales, making results sensitive to a single product cycle. The Company also
faces intense competition, rapid technological change, dependence on continuous
innovation, and execution risk in newer categories. Offsetting this, the growing,
high-margin Services segment diversifies the revenue mix.
*Severity: Moderate · Confidence: High.*

## Key Findings

1. **Financial strength is exceptional** — $162.1B liquidity, $110.5B operating
   cash flow, and 44.1% gross margin leave ample solvency headroom.
2. **Concentration is the binding risk** — iPhone (~52% of sales) and
   China-based assembly are the two exposures most capable of moving results.
3. **Regulatory pressure is rising** — DMA/App Store actions could erode
   Services economics over the medium term.
4. **Services is the structural offset** — 70.8% gross margin and ~9% growth
   steadily de-risk the product-cycle dependence.
5. **No credit/liquidity red flags** — receivables are diversified and debt is
   well-laddered.

## Recommendation

**PROCEED — favorable.** Apple presents a low solvency-risk, high-quality profile
suitable for a core position. Diligence should monitor (a) regulatory outcomes
affecting App Store/Services margins, (b) progress on supply-chain geographic
diversification, and (c) the pace of Services growth relative to iPhone
dependence. No findings warrant withholding from further evaluation.

## Appendix: Data Sources

- **Primary source:** Apple Inc. Form 10-K, FY ended September 30, 2023
  (SEC EDGAR), Items 1, 1A (Risk Factors), 7 (MD&A), 7A, and 8 (Financial
  Statements).
- **Retrieval:** top-5 ChromaDB chunks per risk category
  (`text-embedding-3-small`).
- **Verification:** Critic node (GPT-4o-mini) cross-checked all cited figures
  against retrieved source chunks; self-assessed faithfulness **0.94**.
- *Note: this memo is a packaged DEMO fixture with representative figures for
  illustrating the dashboard; run a live analysis for an authoritative report.*
"""

"""
DDDS - Risk Topic Extraction Pipeline
======================================
Assigns one of 12 predefined risk topics (or 'other') to every sentence
in the passages CSV produced by Run 1 (extract_sections.py).

This script sits between Run 1 (section extraction) and Run 2 (vague /
complex labelling) in the pipeline. Its output feeds directly into the
Neo4j graph as the 'risk_category' field on RiskFactor nodes.

Academic grounding
------------------
Topic taxonomy derived from:
    - Bushee, Gow & Taylor (2018) — obfuscation vs complexity framework
    - SEC Regulation S-K Item 105 (risk factor requirements)
    - Li (2008) — obfuscation precedes earnings deterioration
    - White & Case 2025 emerging risk taxonomy

Pipeline position
-----------------
    run_0 (EDGAR download)
        → run_1 (section extraction → passages.csv)
            → **run_1b (topic extraction → topic_labelled.csv)**  ← YOU ARE HERE
                → run_2 (vague/complex labelling)
                    → run_5 (FinBERT training)

What this script does
---------------------
1. Reads the passages CSV from Run 1
2. Splits each passage into individual sentences
3. Numbers sentences sequentially per filing+section
4. Sends batches of 20 sentences (with 2-sentence overlap for context)
   to GPT-4o-mini for topic classification
5. Each sentence is assigned one of 12 risk topics or 'other'
6. Outputs a sentence-level CSV with topic labels

Batching strategy
-----------------
20 sentences per API call with a 2-sentence context window on each side.
The model sees 24 sentences but only labels the middle 20. This gives the
model surrounding context without overwhelming it — GPT-4o-mini drifts on
long inputs (100+ sentences) but handles 24 sentences reliably.

Modes
-----
    --mode sync     50-sentence test run (synchronous, for prompt validation)
    --mode batch    Full run via OpenAI Batch API (for production)

Usage
-----
    python run_1b_-_topic_extraction.py --input data/passages.csv --mode sync --test-size 50
    python run_1b_-_topic_extraction.py --input data/passages.csv --mode batch

Output CSV columns
------------------
    sentence_id    : unique ID (format: {ticker}_{filing_type}_{section}_{sentence_num})
    identifier     : company ticker or CIK
    company        : company name
    filename       : source filing filename
    filing_type    : 10-K | 10-Q
    section        : MD&A | Risk Factors
    sentence_num   : sentence number within the filing+section (0-indexed)
    text           : sentence text
    topic_label    : one of the 12 topic codes or 'other'
    topic_confidence : high | medium | low
    topic_reason   : one-sentence explanation from GPT-4o-mini
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL           = "gpt-4o-mini"
INPUT_CSV       = "data/passages.csv"
OUTPUT_DIR      = "data/labelled/topics"
BATCH_SIZE      = 20        # sentences per API call (labelled)
CONTEXT_WINDOW  = 2         # sentences of context shown on each side
TEMPERATURE     = 0.0       # deterministic outputs
MAX_RETRIES     = 3
RETRY_DELAY     = 5         # seconds
TEST_SIZE       = 50        # sentences for sync test mode
SEED            = 42
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — RISK TOPIC EXTRACTION
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert financial analyst specialising in SEC disclosure quality assessment for US Industrial companies (SIC 3400–3599: Fabricated Metal Products and Industrial & Commercial Machinery).

YOUR TASK
---------
You will receive a numbered list of sentences extracted from a company's SEC filing (10-K or 10-Q). Each sentence comes from either the Risk Factors or MD&A section.

For each sentence, assign exactly ONE primary risk topic label from the 12 categories below, or assign 'other' if the sentence does not fit any category.

You will see some sentences marked [CONTEXT ONLY] — these are provided for surrounding context to help you classify ambiguous sentences. Do NOT label [CONTEXT ONLY] sentences. Only label sentences marked [LABEL THIS].

TOPIC LABELS AND DEFINITIONS
-----------------------------

1. liquidity_solvency
   Going concern warnings or auditor doubt about continuing operations. Cash burn rate, insufficient liquidity to meet obligations. Covenant breach, waiver requests, or leverage ratio violations. Working capital deficits or current ratio deterioration. Inability to access credit facilities or revolving credit constraints. Near-term debt maturities the company may not be able to refinance.
   BOUNDARY: Long-term capital structure strategy belongs in debt_financing. General macroeconomic uncertainty affecting cash flows belongs in geopolitical_macro. Pension obligations and long-term liabilities belong in financial_sustainability.
   EXAMPLE IN: "The Company may not be able to generate sufficient cash from operations or have access to other sources of liquidity to sustain its operating needs or to meet its obligations as they become due over the next twelve-month period, raising substantial doubt as to the Company's ability to continue as a going concern."
   EXAMPLE OUT: "We intend to refinance our 2026 Senior Notes with proceeds from a new term loan facility." → debt_financing

2. debt_financing
   High leverage ratios and net debt levels. Refinancing risk — upcoming maturities and ability to refinance. Interest rate exposure and rising cost of debt. Ability to raise equity or debt capital. Debt covenant restrictions limiting operational flexibility. Credit rating changes or risk of downgrade. Debt-funded acquisitions and associated integration risk.
   BOUNDARY: Immediate inability to meet debt payments belongs in liquidity_solvency. Working capital and day-to-day cash management belongs in liquidity_solvency. Pension funding obligations belong in financial_sustainability.
   EXAMPLE IN: "Our substantial indebtedness could adversely affect our financial health and prevent us from fulfilling our obligations. As of December 31, 2023, we had approximately $2.1 billion of total indebtedness outstanding."
   EXAMPLE OUT: "We had cash of $450 million and $200 million available under our revolving credit facility as of year end." → liquidity_solvency (positive liquidity disclosure)

3. cost_margin_pressure
   Steel, aluminium, copper and other raw material cost increases. Energy cost inflation affecting manufacturing operations. Wage inflation and rising labour costs. Freight, logistics and transportation cost increases. Inability to pass cost increases through to customers via price. Restructuring charges and cost reduction programme disclosures. Margin compression language — gross margin, EBITDA margin deterioration.
   BOUNDARY: Supply disruption causing production stoppage belongs in supply_chain. Revenue decline from demand weakness belongs in revenue_future_performance. Tariff-related cost increases with geopolitical framing belong in geopolitical_macro.
   EXAMPLE IN: "Significant increases in the cost of steel, aluminium, and other raw materials have adversely affected our gross margins. We have been unable to fully offset these cost increases through price adjustments due to competitive market conditions."
   EXAMPLE OUT: "We implemented a workforce reduction affecting approximately 1,200 employees as part of our strategic restructuring." → operational_product (unless margin impact is explicitly discussed)

4. supply_chain
   Single-source or sole-source supplier dependencies. Component or raw material shortages causing production delays. Extended supplier lead times and inventory build-up. Supplier financial instability creating supply risk. Nearshoring or reshoring cost impacts. Just-in-time inventory model vulnerability disclosures. Logistics network disruption — port congestion, freight availability.
   BOUNDARY: Raw material price increases without supply disruption context belong in cost_margin_pressure. Tariff and trade policy framing belongs in geopolitical_macro. Customer concentration risk belongs in revenue_future_performance.
   EXAMPLE IN: "We rely on a limited number of third-party suppliers for certain key components. The loss of any of these suppliers, or a significant reduction in product availability from them, could disrupt our production and materially adversely affect our results of operations."
   EXAMPLE OUT: "Steel prices increased 35% year-over-year, adversely impacting our cost of goods sold." → cost_margin_pressure

5. revenue_future_performance
   Revenue decline guidance or softening demand signals. Backlog reduction or order cancellation disclosures. Customer concentration — significant portion of revenue from few customers. Market share loss or competitive pricing pressure. Guidance revisions downward — lowered revenue or earnings forecasts. Cyclical demand sensitivity disclosures. Contract renewal risk for long-term customer agreements.
   BOUNDARY: Cost-side drivers of earnings decline belong in cost_margin_pressure. New product development risk belongs in operational_product. Macroeconomic demand drivers belong in geopolitical_macro.
   EXAMPLE IN: "Management anticipates total revenue to decline approximately 10% year-over-year on a constant currency basis for fiscal year 2024, reflecting continued softness in equipment demand across our Americas segment."
   EXAMPLE OUT: "We face significant competition in all of our markets." → other (too generic, no specific revenue impact discussed)

6. geopolitical_macro
   Tariff and trade policy risk — US/China, Section 301/232 tariffs. Sanctions exposure — Russia, Iran, other sanctioned jurisdictions. Foreign currency translation and transaction risk. Inflation impact on operations and purchasing power. Interest rate sensitivity on variable rate debt. Geopolitical conflict impact on markets or supply chains. Recession risk and sensitivity to economic cycles.
   BOUNDARY: Supply chain disruptions from geopolitical causes — classify by primary effect: if supply is disrupted → supply_chain; if costs rise → cost_margin_pressure. Debt cost increases from rate rises → debt_financing. Specific country regulatory risk → regulatory_legal.
   EXAMPLE IN: "The ongoing conflict in Ukraine and associated sanctions have disrupted our operations in Eastern Europe and created significant uncertainty regarding our ability to source certain components from affected regions."
   EXAMPLE OUT: "We operate in highly competitive markets." → other (not geopolitical)

7. regulatory_legal
   Environmental remediation liability and EPA compliance risk. SEC investigation or enforcement action disclosure. Pending or threatened material litigation. Product liability claims and associated contingent liabilities. Government contract compliance risk — FCPA, export controls. Antitrust investigations or market conduct risk. New regulation increasing compliance costs.
   BOUNDARY: Cybersecurity regulation belongs in cybersecurity_technology. Employment law and labour regulation belongs in human_capital. Only label if specific exposure, liability, or investigation is described — generic "we are subject to various regulations" is 'other'.
   EXAMPLE IN: "The SEC has issued a formal order of investigation related to our revenue recognition practices. We are cooperating fully with the investigation, but cannot predict its outcome or the potential financial impact at this time."
   EXAMPLE OUT: "We are subject to various environmental laws and regulations." → other (too generic)

8. operational_product
   Product recalls and associated costs. Warranty claims increasing beyond historical rates. Quality control failures in manufacturing. Plant or facility operational disruptions. IT system failures affecting operations (non-cyber). Workforce reduction and operational restructuring impacts. New product launch delays or performance shortfalls.
   BOUNDARY: Cybersecurity attacks on IT systems belong in cybersecurity_technology. Supply chain disruptions causing production stoppage belong in supply_chain. Product liability litigation belongs in regulatory_legal. Competitive risk with no specific product quality issue is 'other'.
   EXAMPLE IN: "We have experienced an increase in warranty claims related to certain products manufactured at our Ohio facility. We have accrued additional warranty reserves of $24 million in the current year to address these claims."
   EXAMPLE OUT: "We face competition from lower-cost manufacturers." → other (competitive risk, not product quality)

9. cybersecurity_technology
   Cybersecurity incidents — past breaches or current vulnerability disclosure. Ransomware risk and operational disruption from cyber attacks. Data privacy and customer data protection obligations. IT system modernisation risk and legacy technology exposure. Third-party technology provider dependency and associated risk. AI adoption risk — model reliability, data integrity, regulatory exposure. Technology obsolescence threatening competitive position.
   BOUNDARY: IT operational disruptions without cyber cause belong in operational_product. Cybersecurity regulation compliance cost belongs in regulatory_legal. General digital transformation strategy — only label here if failure risk is explicitly discussed.
   EXAMPLE IN: "In fiscal 2023, we experienced a ransomware attack that disrupted our manufacturing operations for approximately 12 days and resulted in costs of approximately $8 million to remediate and restore systems."
   EXAMPLE OUT: "We have invested significantly in digital transformation initiatives." → other (positive framing, no failure risk discussed)

10. human_capital
    Key executive departures or CEO/CFO succession risk. Inability to attract or retain skilled labour in tight market. Labour relations — union strikes, work stoppages, contract negotiations. Workforce reduction announcements and associated restructuring charges. Dependence on specific individuals for key customer relationships. Wage inflation and rising total compensation costs (when framed as workforce risk, not margin pressure).
    BOUNDARY: Wage cost as a margin driver without workforce risk context belongs in cost_margin_pressure. Employment law and regulatory compliance belongs in regulatory_legal.
    EXAMPLE IN: "The departure of our Chief Executive Officer in February 2024 created significant uncertainty during our strategic planning process. We have retained an executive search firm to identify a permanent successor."
    EXAMPLE OUT: "We employed approximately 14,000 people as of December 31, 2023." → other (factual disclosure, not a risk sentence)

11. financial_sustainability
    Pension obligation funding gaps and actuarial assumption changes. Goodwill and intangible asset impairment risk and charges. Dividend sustainability and ability to maintain distributions. Capital expenditure requirements relative to cash generation. Asset write-downs and impairment charges. Off-balance sheet obligations and contingent liabilities. Defined benefit plan underfunding disclosures.
    BOUNDARY: Immediate debt maturity pressure belongs in liquidity_solvency. Long-term debt structure belongs in debt_financing. Revenue sustainability belongs in revenue_future_performance.
    EXAMPLE IN: "Based on the results of our annual impairment test, the carrying value of the Americas Manufacturing reporting unit exceeded its fair value, resulting in a non-cash impairment charge of $171.9 million in the fourth quarter of 2022."
    EXAMPLE OUT: "We contributed $45 million to our pension plan during 2023." → other (routine contribution; only label here if funding gap or obligation growth is discussed)

12. accounting_audit
    Material weakness in internal controls over financial reporting. Restatement of prior period financial statements. Changes in critical accounting estimates or policies. Auditor change disclosure. Revenue recognition complexity or judgement disclosure. Going concern qualification from auditors. SEC comment letter on accounting practices.
    BOUNDARY: General description of accounting policies with no risk framing is 'other'. Tax accounting changes belong here only if related to a restatement. SEC investigation for non-accounting reasons belongs in regulatory_legal.
    EXAMPLE IN: "The Audit Committee concluded that the Company will restate its previously issued consolidated financial statements to correct its accounting with respect to its Equity Units in response to comments received from the Securities and Exchange Commission."
    EXAMPLE OUT: "We recognize revenue in accordance with ASC 606." → other (standard policy statement; only label here if complexity, judgement, or risk is explicitly discussed)

CLASSIFICATION RULES
--------------------
1. Assign ONE label only — choose the most specific applicable topic.
2. If a sentence discusses BOTH cost increases AND supply disruption, choose the primary cause:
   - If the sentence focuses on WHY supply was disrupted → supply_chain
   - If the sentence focuses on HOW MUCH costs increased → cost_margin_pressure
3. Going concern language ALWAYS gets liquidity_solvency regardless of underlying cause.
4. Generic risk boilerplate with no company-specific content → 'other'.
   Examples: "we face competition in all markets", "results may differ materially",
   "we are subject to various laws and regulations"
5. Factual statements with no risk framing → 'other'.
   Examples: "we employed 14,000 people", "we recognize revenue under ASC 606",
   "we have offices in 12 countries"
6. Navigational, transitional, or heading sentences → 'other'.
   Examples: "see Note 8 for additional details", "the following table summarises..."
7. When in doubt between two topics, choose the one where the sentence would be
   MOST USEFUL to a buy-side equity analyst assessing near-term financial risk.

OUTPUT FORMAT
-------------
Respond ONLY with a valid JSON object containing a "results" array.
Each element must have the sentence number, label, confidence, and a one-sentence reason.

Do NOT label sentences marked [CONTEXT ONLY]. Only label sentences marked [LABEL THIS].

Example output:
{
  "results": [
    {"sentence": 5, "label": "cost_margin_pressure", "confidence": "high", "reason": "Discusses raw material cost increases and their impact on gross margins with specific figures."},
    {"sentence": 6, "label": "supply_chain", "confidence": "medium", "reason": "References single-source supplier dependency creating production risk."},
    {"sentence": 7, "label": "other", "confidence": "high", "reason": "Generic boilerplate risk disclaimer with no company-specific content."}
  ]
}

CRITICAL REMINDERS
------------------
- You MUST return a result for EVERY sentence marked [LABEL THIS]. Do not skip any.
- The sentence numbers in your output MUST exactly match the sentence numbers in the input.
- If you are unsure, your best guess with confidence "low" is ALWAYS better than skipping the sentence.
- 'other' is a valid and expected label — approximately 15-25% of sentences in typical filings are boilerplate, navigational, or factual statements that do not fit any risk topic.
"""

# ---------------------------------------------------------------------------
# USER PROMPT TEMPLATE
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """Company: {company} ({identifier})
Filing: {filing_type} — Section: {section}

Classify each sentence marked [LABEL THIS] into one of the 12 risk topics or 'other'.
Sentences marked [CONTEXT ONLY] are provided for surrounding context — do NOT label them.

---SENTENCES---
{sentences_block}
---END---"""


# ---------------------------------------------------------------------------
# VALID TOPIC LABELS
# ---------------------------------------------------------------------------
VALID_LABELS = {
    "liquidity_solvency",
    "debt_financing",
    "cost_margin_pressure",
    "supply_chain",
    "revenue_future_performance",
    "geopolitical_macro",
    "regulatory_legal",
    "operational_product",
    "cybersecurity_technology",
    "human_capital",
    "financial_sustainability",
    "accounting_audit",
    "other",
}


# ---------------------------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """
    Splits a passage into individual sentences using regex-based rules.
    Handles common SEC filing patterns (e.g. dollar amounts with periods,
    abbreviations like 'Inc.', 'Corp.', 'Ltd.', numbered items).

    Returns a list of sentence strings with whitespace normalised.
    """
    # Protect common abbreviations from being treated as sentence boundaries
    protected = text
    abbreviations = [
        "Inc.", "Corp.", "Ltd.", "Co.", "L.P.", "LLC.", "N.A.", "S.A.",
        "Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "Prof.",
        "No.", "Nos.", "Vol.", "vs.", "etc.", "approx.", "i.e.", "e.g.",
        "U.S.", "U.K.", "E.U.",
    ]
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", "<<DOT>>"))

    # Protect decimal numbers (e.g. $14.7, 3.5%, 10.2)
    protected = re.sub(r"(\d)\.(\d)", r"\1<<DOT>>\2", protected)

    # Split on sentence-ending punctuation followed by whitespace and uppercase
    # or followed by whitespace and a number (common in SEC filings)
    raw_sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\(])',
        protected
    )

    # Restore protected dots
    sentences = [s.replace("<<DOT>>", ".").strip() for s in raw_sentences]

    # Filter out empty or very short fragments (< 15 chars)
    sentences = [s for s in sentences if len(s) >= 15]

    return sentences


# ---------------------------------------------------------------------------
# BATCH CONSTRUCTION
# ---------------------------------------------------------------------------

def build_batches(
    sentences: list[str],
    batch_size: int = BATCH_SIZE,
    context_window: int = CONTEXT_WINDOW,
) -> list[dict]:
    """
    Groups sentences into batches of `batch_size` for API calls.
    Each batch includes `context_window` sentences of overlap on each side,
    marked as [CONTEXT ONLY] in the prompt.

    Returns a list of dicts, each containing:
        - 'label_range': (start_idx, end_idx) of sentences to label (inclusive)
        - 'sentences_block': formatted string for the prompt
        - 'expected_ids': list of sentence numbers that must appear in the response
    """
    total = len(sentences)
    batches = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)  # exclusive end of label range

        # Context boundaries
        ctx_start = max(0, start - context_window)
        ctx_end = min(total, end + context_window)

        # Build the formatted sentences block
        lines = []
        expected_ids = []

        for i in range(ctx_start, ctx_end):
            if i < start or i >= end:
                # Context sentence — shown but not labelled
                lines.append(f"[CONTEXT ONLY] Sentence {i}: {sentences[i]}")
            else:
                # Target sentence — must be labelled
                lines.append(f"[LABEL THIS] Sentence {i}: {sentences[i]}")
                expected_ids.append(i)

        batches.append({
            "label_range": (start, end - 1),
            "sentences_block": "\n\n".join(lines),
            "expected_ids": expected_ids,
        })

    return batches


# ---------------------------------------------------------------------------
# API CALLS
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    """Initialises the OpenAI client from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    return OpenAI(api_key=api_key)


def call_gpt4o_mini(
    client: OpenAI,
    company: str,
    identifier: str,
    filing_type: str,
    section: str,
    sentences_block: str,
    expected_ids: list[int],
) -> list[dict] | None:
    """
    Single synchronous API call to GPT-4o-mini for topic extraction.

    Args:
        client:           OpenAI client instance
        company:          Company name for context
        identifier:       Ticker / CIK
        filing_type:      10-K or 10-Q
        section:          MD&A or Risk Factors
        sentences_block:  Formatted sentences string
        expected_ids:     List of sentence numbers that MUST appear in the response

    Returns:
        List of result dicts or None on total failure.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        company=company,
        identifier=identifier,
        filing_type=filing_type,
        section=section,
        sentences_block=sentences_block,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # Validate structure
            if "results" not in parsed:
                raise ValueError("Response missing 'results' key")

            results = parsed["results"]

            # Validate all expected sentence IDs are present
            returned_ids = {r["sentence"] for r in results}
            missing = set(expected_ids) - returned_ids
            if missing:
                raise ValueError(
                    f"Response missing sentence IDs: {sorted(missing)}. "
                    f"Got: {sorted(returned_ids)}. Expected: {sorted(expected_ids)}"
                )

            # Validate labels
            for r in results:
                if r["label"] not in VALID_LABELS:
                    print(f"    [!] Invalid label '{r['label']}' for sentence {r['sentence']} — setting to 'other'")
                    r["label"] = "other"

            # Filter to only expected IDs (ignore any context sentences
            # the model may have labelled despite instructions)
            results = [r for r in results if r["sentence"] in set(expected_ids)]

            return results

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [!] API call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    [!] All retries failed: {e}")
                return None


# ---------------------------------------------------------------------------
# BATCH API SUPPORT
# ---------------------------------------------------------------------------

def prepare_batch_file(
    all_requests: list[dict],
    output_path: str,
) -> str:
    """
    Writes a JSONL file formatted for the OpenAI Batch API.

    Each line is a request object with:
        - custom_id: unique identifier for matching responses
        - method: POST
        - url: /v1/chat/completions
        - body: the chat completion request

    Returns the path to the JSONL file.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for req in all_requests:
            line = json.dumps(req, ensure_ascii=False)
            f.write(line + "\n")

    print(f"  Batch file written: {output_path}")
    print(f"  Total requests:     {len(all_requests)}")
    return output_path


def submit_batch(client: OpenAI, batch_file_path: str) -> str:
    """
    Uploads the JSONL file and submits it as a Batch API job.
    Returns the batch job ID for status checking.
    """
    # Upload the file
    with open(batch_file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    print(f"  File uploaded: {uploaded.id}")

    # Create the batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "DDDS topic extraction"},
    )

    print(f"  Batch submitted: {batch.id}")
    print(f"  Status:           {batch.status}")
    return batch.id


def check_batch_status(client: OpenAI, batch_id: str) -> dict:
    """Returns the current status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def download_batch_results(client: OpenAI, batch_id: str, output_path: str) -> list[dict]:
    """
    Downloads and parses results from a completed batch job.
    Returns a list of (custom_id, results) tuples.
    """
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch_id} is not completed. Status: {batch.status}"
        )

    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output file.")

    # Download the output file
    content = client.files.content(batch.output_file_id)
    raw_text = content.text

    # Save raw output for debugging
    raw_path = output_path.replace(".csv", "_batch_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"  Raw batch output saved: {raw_path}")

    # Parse each line
    all_results = []
    for line in raw_text.strip().split("\n"):
        obj = json.loads(line)
        custom_id = obj["custom_id"]

        if obj["response"]["status_code"] != 200:
            print(f"    [!] Request {custom_id} failed: {obj['response']}")
            continue

        body = obj["response"]["body"]
        raw_content = body["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(raw_content)
            results = parsed.get("results", [])
            all_results.append({"custom_id": custom_id, "results": results})
        except json.JSONDecodeError as e:
            print(f"    [!] JSON parse failed for {custom_id}: {e}")

    # Check for errors file
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = output_path.replace(".csv", "_batch_errors.jsonl")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_content.text)
        print(f"  Errors file saved: {error_path}")

    return all_results


# ---------------------------------------------------------------------------
# DATA LOADING & SENTENCE PREPARATION
# ---------------------------------------------------------------------------

def load_and_prepare(input_path: str) -> pd.DataFrame:
    """
    Loads the passages CSV from Run 1, splits passages into individual
    sentences, and numbers them sequentially per filing+section.

    Returns a DataFrame with one row per sentence.
    """
    df = pd.read_csv(input_path)

    required = {"text", "identifier", "filename", "filing_type", "section", "company"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Ensure you are using the output of run_1_-_risk_factor_script.py"
        )

    # Drop non-filing rows (only 10-K and 10-Q)
    df = df[df["filing_type"].isin(["10-K", "10-Q"])].copy()
    df = df.dropna(subset=["text", "identifier"])
    df["text"] = df["text"].astype(str).str.strip()
    df["identifier"] = df["identifier"].astype(str).str.strip()

    # Group by filing + section and split into sentences
    sentence_rows = []

    groups = df.groupby(["identifier", "filename", "filing_type", "section", "company"])

    for (identifier, filename, filing_type, section, company), group_df in groups:
        # Concatenate all passages for this filing+section
        full_text = " ".join(group_df["text"].tolist())
        sentences = split_into_sentences(full_text)

        for i, sent in enumerate(sentences):
            # Create a deterministic sentence ID
            sent_id = f"{identifier}_{filing_type}_{section}_{i:04d}"

            sentence_rows.append({
                "sentence_id":  sent_id,
                "identifier":   identifier,
                "company":      company,
                "filename":     filename,
                "filing_type":  filing_type,
                "section":      section,
                "sentence_num": i,
                "text":         sent,
            })

    result = pd.DataFrame(sentence_rows)
    print(f"  {len(result)} sentences extracted from {len(df)} passages")
    print(f"  {result['identifier'].nunique()} companies")
    print(f"  {result.groupby(['identifier', 'filename', 'section']).ngroups} filing-sections")
    return result


# ---------------------------------------------------------------------------
# SYNC MODE — SMALL TEST RUN
# ---------------------------------------------------------------------------

def run_sync(
    sentence_df: pd.DataFrame,
    client: OpenAI,
    test_size: int = TEST_SIZE,
) -> pd.DataFrame:
    """
    Runs topic extraction synchronously on a small test sample.
    Used for prompt validation before committing to a full Batch API run.
    """
    # Sample sentences — take the first `test_size` from a random filing
    np.random.seed(SEED)
    groups = list(sentence_df.groupby(["identifier", "filename", "section"]))
    np.random.shuffle(groups)

    # Collect enough sentences for the test
    test_sentences = []
    for (identifier, filename, section), group_df in groups:
        test_sentences.extend(group_df.to_dict("records"))
        if len(test_sentences) >= test_size:
            break
    test_sentences = test_sentences[:test_size]

    if not test_sentences:
        raise ValueError("No sentences available for testing.")

    test_df = pd.DataFrame(test_sentences)
    print(f"  Test sample: {len(test_df)} sentences")

    # Process by filing+section group
    all_results = []
    groups = test_df.groupby(["identifier", "filename", "section"])

    for (identifier, filename, section), group_df in groups:
        company = group_df.iloc[0]["company"]
        filing_type = group_df.iloc[0]["filing_type"]
        sentences = group_df["text"].tolist()
        sentence_nums = group_df["sentence_num"].tolist()

        print(f"\n  {identifier} | {filing_type} | {section} | {len(sentences)} sentences")

        # Build batches with context
        batches = build_batches(sentences)

        for batch_idx, batch in enumerate(batches):
            # Remap sentence numbers to match the original numbering
            expected_ids_remapped = []
            for local_id in batch["expected_ids"]:
                if local_id < len(sentence_nums):
                    expected_ids_remapped.append(sentence_nums[local_id])

            # Rebuild sentences block with original numbering
            lines = batch["sentences_block"].split("\n\n")
            remapped_lines = []
            for line in lines:
                # Replace local sentence numbers with original
                match = re.match(r"(\[(?:CONTEXT ONLY|LABEL THIS)\]) Sentence (\d+): (.+)", line)
                if match:
                    tag = match.group(1)
                    local_num = int(match.group(2))
                    text = match.group(3)
                    if local_num < len(sentence_nums):
                        orig_num = sentence_nums[local_num]
                        remapped_lines.append(f"{tag} Sentence {orig_num}: {text}")
                    else:
                        remapped_lines.append(line)
                else:
                    remapped_lines.append(line)

            remapped_block = "\n\n".join(remapped_lines)

            print(f"    Batch {batch_idx+1}: sentences {batch['label_range'][0]}-{batch['label_range'][1]}", end=" ... ")

            results = call_gpt4o_mini(
                client=client,
                company=company,
                identifier=identifier,
                filing_type=filing_type,
                section=section,
                sentences_block=remapped_block,
                expected_ids=expected_ids_remapped,
            )

            if results:
                for r in results:
                    r["identifier"] = identifier
                    r["filename"] = filename
                    r["section"] = section
                all_results.extend(results)
                print(f"✓ {len(results)} labelled")
            else:
                print("✗ FAILED")

            time.sleep(1)  # rate limit respect

    # Merge results back with sentence data
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("\n  [!] No results returned. Check API connection and prompt.")
        return test_df

    results_df = results_df.rename(columns={
        "sentence": "sentence_num",
        "label": "topic_label",
        "confidence": "topic_confidence",
        "reason": "topic_reason",
    })

    merged = test_df.merge(
        results_df[["sentence_num", "identifier", "filename", "section",
                     "topic_label", "topic_confidence", "topic_reason"]],
        on=["sentence_num", "identifier", "filename", "section"],
        how="left",
    )

    return merged


# ---------------------------------------------------------------------------
# BATCH MODE — FULL PRODUCTION RUN
# ---------------------------------------------------------------------------

def run_batch_prepare(
    sentence_df: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Prepares the JSONL file for the OpenAI Batch API.
    Returns the path to the batch file.
    """
    all_requests = []
    request_idx = 0

    groups = sentence_df.groupby(["identifier", "filename", "section"])

    for (identifier, filename, section), group_df in groups:
        company = group_df.iloc[0]["company"]
        filing_type = group_df.iloc[0]["filing_type"]
        sentences = group_df["text"].tolist()
        sentence_nums = group_df["sentence_num"].tolist()

        batches = build_batches(sentences)

        for batch in batches:
            # Remap sentence numbers to originals
            lines = batch["sentences_block"].split("\n\n")
            remapped_lines = []
            expected_ids_remapped = []

            for line in lines:
                match = re.match(r"(\[(?:CONTEXT ONLY|LABEL THIS)\]) Sentence (\d+): (.+)", line)
                if match:
                    tag = match.group(1)
                    local_num = int(match.group(2))
                    text = match.group(3)
                    if local_num < len(sentence_nums):
                        orig_num = sentence_nums[local_num]
                        remapped_lines.append(f"{tag} Sentence {orig_num}: {text}")
                        if tag == "[LABEL THIS]":
                            expected_ids_remapped.append(orig_num)
                    else:
                        remapped_lines.append(line)
                else:
                    remapped_lines.append(line)

            remapped_block = "\n\n".join(remapped_lines)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                company=company,
                identifier=identifier,
                filing_type=filing_type,
                section=section,
                sentences_block=remapped_block,
            )

            # Custom ID encodes metadata for response matching
            custom_id = f"{identifier}__{filename}__{section}__{request_idx:05d}"
            # Sanitise custom_id (no spaces or special chars)
            custom_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", custom_id)

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
            }

            all_requests.append(request)
            request_idx += 1

    batch_path = os.path.join(output_dir, "topic_extraction_batch.jsonl")
    return prepare_batch_file(all_requests, batch_path)


# ---------------------------------------------------------------------------
# OUTPUT EXPORT
# ---------------------------------------------------------------------------

def export_results(merged_df: pd.DataFrame, output_dir: str):
    """Saves the topic-labelled sentence CSV and prints summary statistics."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "topic_labelled.csv")
    merged_df.to_csv(output_path, index=False)

    # Summary
    labelled = merged_df[merged_df["topic_label"].notna()]
    unlabelled = merged_df[merged_df["topic_label"].isna()]

    print(f"\n{'='*60}")
    print(f"  Topic Extraction Summary")
    print(f"{'='*60}")
    print(f"  Total sentences:     {len(merged_df)}")
    print(f"  Labelled:            {len(labelled)}")
    print(f"  Unlabelled (failed): {len(unlabelled)}")

    if len(labelled) > 0:
        print(f"\n  Topic distribution:")
        topic_counts = labelled["topic_label"].value_counts()
        for topic, count in topic_counts.items():
            pct = 100 * count / len(labelled)
            print(f"    {topic:<35} {count:>5}  ({pct:>5.1f}%)")

        print(f"\n  Confidence distribution:")
        conf_counts = labelled["topic_confidence"].value_counts()
        for conf, count in conf_counts.items():
            print(f"    {conf:<10} {count:>5}")

    print(f"\n  Output: {output_path}")
    print(f"{'='*60}")

    return output_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DDDS Risk Topic Extraction Pipeline"
    )
    parser.add_argument("--input", default=INPUT_CSV,
                        help="Path to passages CSV from Run 1")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for output files")
    parser.add_argument("--mode", choices=["sync", "batch", "batch-status", "batch-download"],
                        default="sync",
                        help="sync: small test run | batch: prepare & submit Batch API | "
                             "batch-status: check job status | batch-download: download results")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE,
                        help="Number of sentences for sync test mode")
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Batch job ID for status/download commands")
    args = parser.parse_args()

    print(f"\n[DDDS] Risk Topic Extraction Pipeline")
    print(f"  Input:      {args.input}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Mode:       {args.mode}")
    print(f"  Model:      {MODEL}\n")

    # ------------------------------------------------------------------
    # SYNC MODE — test run
    # ------------------------------------------------------------------
    if args.mode == "sync":
        print("[1/4] Loading and preparing sentences...")
        sentence_df = load_and_prepare(args.input)

        print(f"\n[2/4] Connecting to OpenAI...")
        client = build_client()
        print("  Connected")

        print(f"\n[3/4] Running sync test ({args.test_size} sentences)...")
        results_df = run_sync(sentence_df, client, args.test_size)

        print(f"\n[4/4] Exporting results...")
        export_results(results_df, args.output_dir)

        print("\n[Done] Sync test complete.")
        print("  Review the output CSV. If topic distribution looks reasonable,")
        print("  run with --mode batch for the full production run.")

    # ------------------------------------------------------------------
    # BATCH MODE — prepare and submit
    # ------------------------------------------------------------------
    elif args.mode == "batch":
        print("[1/3] Loading and preparing sentences...")
        sentence_df = load_and_prepare(args.input)

        print(f"\n[2/3] Preparing Batch API file...")
        batch_path = run_batch_prepare(sentence_df, args.output_dir)

        print(f"\n[3/3] Submitting to OpenAI Batch API...")
        client = build_client()
        batch_id = submit_batch(client, batch_path)

        print(f"\n[Done] Batch submitted.")
        print(f"  Batch ID: {batch_id}")
        print(f"  Check status:   python {Path(__file__).name} --mode batch-status --batch-id {batch_id}")
        print(f"  Download:       python {Path(__file__).name} --mode batch-download --batch-id {batch_id}")

        # Save the sentence_df for later matching
        sentence_path = os.path.join(args.output_dir, "sentences_prepared.csv")
        sentence_df.to_csv(sentence_path, index=False)
        print(f"  Sentence data saved: {sentence_path}")

    # ------------------------------------------------------------------
    # BATCH STATUS — check job
    # ------------------------------------------------------------------
    elif args.mode == "batch-status":
        if not args.batch_id:
            raise ValueError("--batch-id required for batch-status mode")

        client = build_client()
        status = check_batch_status(client, args.batch_id)

        print(f"  Batch ID:    {status['id']}")
        print(f"  Status:      {status['status']}")
        print(f"  Requests:    {status['request_counts']['completed']}/{status['request_counts']['total']} completed")
        if status["request_counts"]["failed"]:
            print(f"  Failed:      {status['request_counts']['failed']}")

    # ------------------------------------------------------------------
    # BATCH DOWNLOAD — retrieve results
    # ------------------------------------------------------------------
    elif args.mode == "batch-download":
        if not args.batch_id:
            raise ValueError("--batch-id required for batch-download mode")

        client = build_client()

        print("[1/3] Downloading batch results...")
        output_path = os.path.join(args.output_dir, "topic_labelled.csv")
        batch_results = download_batch_results(client, args.batch_id, output_path)
        print(f"  {len(batch_results)} responses downloaded")

        print("\n[2/3] Loading sentence data...")
        sentence_path = os.path.join(args.output_dir, "sentences_prepared.csv")
        sentence_df = pd.read_csv(sentence_path)

        print("\n[3/3] Merging results...")
        # Flatten all results into a single list with metadata from custom_id
        all_result_rows = []
        for batch_resp in batch_results:
            custom_id = batch_resp["custom_id"]
            # Parse metadata from custom_id: {identifier}__{filename}__{section}__{idx}
            parts = custom_id.split("__")
            identifier = parts[0] if len(parts) > 0 else ""
            filename = parts[1] if len(parts) > 1 else ""
            section = parts[2] if len(parts) > 2 else ""

            for r in batch_resp["results"]:
                all_result_rows.append({
                    "sentence_num": r.get("sentence"),
                    "identifier": identifier,
                    "filename": filename,
                    "section": section,
                    "topic_label": r.get("label", "other"),
                    "topic_confidence": r.get("confidence", "low"),
                    "topic_reason": r.get("reason", ""),
                })

        results_df = pd.DataFrame(all_result_rows)

        # Merge with sentence data
        merged = sentence_df.merge(
            results_df,
            on=["sentence_num", "identifier", "filename", "section"],
            how="left",
        )

        export_results(merged, args.output_dir)

        print("\n[Done] Batch results processed.")
        print("  Next: run_7 (Neo4j graph building) with the topic_labelled.csv")


if __name__ == "__main__":
    main()
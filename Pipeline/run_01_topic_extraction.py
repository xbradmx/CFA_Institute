"""
DDDS - Risk Topic Extraction Pipeline (Production)
====================================================
Walks the company filings folder, extracts Risk Factors and MD&A sections
from each HTML filing, splits into sentences, and sends to GPT-4o-mini
for topic classification via the OpenAI Batch API.

Outputs a master sentence-level CSV with columns that allow downstream
grouping by company, filing, date, section, and topic.

Pipeline position
-----------------
    run_0 (EDGAR download)
        → **run_1 (topic extraction → topic_labelled.csv)**  ← YOU ARE HERE
            → run_6 (FinBERT prediction)
                → run_7 (Neo4j graph building)

Modes
-----
    --mode extract      Extract sections + sentences from HTMLs → sentences CSV
    --mode batch        Prepare + submit Batch API job for topic labelling
    --mode status       Check batch job status
    --mode watch        Auto-poll batch status every 60s, download when complete
    --mode download     Download results + merge into final labelled CSV

Usage
-----
    python "run 1 - Topic Extraction.py" --mode extract
    python "run 1 - Topic Extraction.py" --mode batch
    python "run 1 - Topic Extraction.py" --mode status --batch-id batch_xxx
    python "run 1 - Topic Extraction.py" --mode watch --batch-id batch_xxx
    python "run 1 - Topic Extraction.py" --mode download --batch-id batch_xxx

Output CSV columns
------------------
    sentence_id      : unique ID ({ticker}_{filing_type}_{filing_date}_{section_code}_{num})
    ticker           : company ticker
    filing_type      : 10-K | 10-Q
    filing_date      : YYYY-MM-DD (from filename)
    section          : Risk Factors | MD&A
    sentence_num     : sentence number within filing+section (0-indexed)
    text             : sentence text
    topic_label      : one of 12 topic codes or 'other'
    topic_confidence : high | medium | low
    topic_reason     : one-sentence explanation from GPT-4o-mini
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FILINGS_DIR     = Path("data/Filings by company")
OUTPUT_DIR      = Path("data/labelled/topics")
SENTENCES_CSV   = OUTPUT_DIR / "sentences_extracted.csv"
LABELLED_CSV    = OUTPUT_DIR / "topic_labelled.csv"
BATCH_JSONL     = OUTPUT_DIR / "topic_batch_requests.jsonl"
MODEL           = "gpt-4o-mini"
BATCH_SIZE      = 10
CONTEXT_WINDOW  = 2
MAX_RETRIES     = 3
RETRY_DELAY     = 5
WATCH_INTERVAL  = 60         # seconds between status checks in watch mode
MAX_PER_BATCH   = 15000      # max requests per batch submission (40M token enqueue limit)

VALID_LABELS = {
    "liquidity_solvency", "debt_financing", "cost_margin_pressure",
    "supply_chain", "revenue_future_performance", "geopolitical_macro",
    "regulatory_legal", "operational_product", "cybersecurity_technology",
    "human_capital", "financial_sustainability", "accounting_audit", "other",
}


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert financial analyst classifying sentences from SEC filings (US Industrials, SIC 3400-3599) into risk topics.

For each sentence marked [LABEL THIS], assign exactly ONE topic label from below, or 'other' if it doesn't fit any category. Do NOT label sentences marked [CONTEXT ONLY].

TOPIC LABELS:
1. liquidity_solvency - Going concern, cash burn, covenant breach, working capital deficit
2. debt_financing - Leverage, refinancing risk, interest rate exposure, credit rating
3. cost_margin_pressure - Raw material costs, wage inflation, freight costs, margin compression
4. supply_chain - Supplier concentration, component shortages, logistics disruption
5. revenue_future_performance - Revenue decline, backlog reduction, customer concentration, guidance cuts
6. geopolitical_macro - Tariffs, sanctions, FX risk, inflation, recession sensitivity
7. regulatory_legal - Environmental liability, SEC investigation, material litigation, FCPA
8. operational_product - Product recalls, warranty claims, quality failures, plant disruption
9. cybersecurity_technology - Cyber breaches, ransomware, data privacy, IT modernisation risk
10. human_capital - Executive departures, labour shortages, union disputes, succession risk
11. financial_sustainability - Goodwill impairment, pension gaps, dividend sustainability, capex burden
12. accounting_audit - Material weakness, restatements, auditor changes, revenue recognition complexity
other - Generic boilerplate, navigational sentences, factual statements with no risk framing, generic competition statements without specific revenue/demand/guidance impact

RULES:
- ONE label per sentence. Choose the most specific applicable topic.
- Going concern language ALWAYS gets liquidity_solvency.
- Generic boilerplate ("we face competition", "results may differ materially") -> other.
- Generic competition statements without specific revenue, demand, or guidance impact -> other.
- Factual statements with no risk framing -> other.
- Navigational sentences ("see Note 8", "the following table") -> other.
- Financial table data or numeric fragments -> other.
- ~15-25% of sentences are expected to be 'other'.

OUTPUT FORMAT: Respond ONLY with valid JSON:
{
  "results": [
    {"sentence": 5, "label": "cost_margin_pressure", "confidence": "high", "reason": "Discusses raw material cost increases."},
    ...
  ]
}

You MUST return exactly one result for EVERY sentence marked [LABEL THIS]. The count of results must equal the count of [LABEL THIS] sentences. Never skip any."""


# ---------------------------------------------------------------------------
# HTML PARSING
# ---------------------------------------------------------------------------

def extract_text_from_html(path: Path) -> str:
    """Extracts clean text from an EDGAR HTML filing."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                for tag in soup(["script", "style", "head"]):
                    tag.decompose()
                return soup.get_text(separator="\n")
        except UnicodeDecodeError:
            continue
    return ""


def clean_text(text: str) -> str:
    """Cleans extracted text. Normalises non-breaking spaces first."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*[-\u2013]\s*\d+\s*[-\u2013]\s*$", "", text)
    text = re.sub(r"(?m)^\s*[Pp]age\s+\d+\s*$", "", text)
    text = re.sub(r"[ \t]{3,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# SECTION EXTRACTION (full-text search on collapsed whitespace)
# ---------------------------------------------------------------------------

SECTION_START = {
    "Risk Factors": [
        r"item\s+1a\.?\s*[.\-\u2013\u2014:)]*\s*risk\s+factors",
    ],
    "MD&A": [
        r"item\s+7\.?\s*[.\-\u2013\u2014:)]*\s*management.{1,5}s\s+discussion",
        r"item\s+2\.?\s*[.\-\u2013\u2014:)]*\s*management.{1,5}s\s+discussion",
        r"management.{1,5}s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition",
    ],
}

SECTION_END = {
    "Risk Factors": {
        "10-K": [
            r"item\s+1b\.?\s*[.\-\u2013\u2014:)]*\s*unresolved\s+staff",
            r"item\s+1c\.?\s*[.\-\u2013\u2014:)]*\s*cybersecurity",
            r"item\s+2\.?\s*[.\-\u2013\u2014:)]*\s*properties",
        ],
        "10-Q": [
            r"item\s+2\.?\s*[.\-\u2013\u2014:)]*\s*unregistered\s+sales",
            r"item\s+2\.?\s*[.\-\u2013\u2014:)]*\s*management.s\s+discussion",
        ],
    },
    "MD&A": {
        "10-K": [
            r"item\s+7a\.?\s*[.\-\u2013\u2014:)]*\s*quantitative\s+and\s+qualitative",
        ],
        "10-Q": [
            r"item\s+3\.?\s*[.\-\u2013\u2014:)]*\s*quantitative\s+and\s+qualitative",
        ],
    },
}


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines, \\xa0, \\u200b) into
    single spaces so that headers split across lines become matchable."""
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def extract_section(text: str, section_name: str, filing_type: str) -> str | None:
    """Extracts a named section by searching the full collapsed text.

    1. Collapse all whitespace so split headers become one string
    2. Find ALL candidate start positions via regex
    3. For each start, find the nearest end boundary
    4. Return the longest section found (naturally skips TOC entries)
    """
    start_patterns = SECTION_START.get(section_name, [])
    end_patterns = SECTION_END.get(section_name, {}).get(filing_type, [])

    if not end_patterns:
        return None

    collapsed = collapse_whitespace(text)

    # --- Find all candidate start positions ---
    starts = []
    for pattern in start_patterns:
        for match in re.finditer(pattern, collapsed, re.IGNORECASE):
            starts.append((match.start(), match.end()))

    if not starts:
        return None

    # --- For each start, find the nearest end boundary ---
    best_section = None
    best_length = 0

    for start_pos, header_end in starts:
        search_from = header_end
        end_pos = len(collapsed)

        for end_pattern in end_patterns:
            match = re.search(end_pattern, collapsed[search_from:], re.IGNORECASE)
            if match:
                candidate_end = search_from + match.start()
                if candidate_end < end_pos:
                    end_pos = candidate_end

        section_text = collapsed[start_pos:end_pos].strip()

        if len(section_text) > best_length:
            best_length = len(section_text)
            best_section = section_text

    return best_section


# ---------------------------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences. Filters table data and short fragments."""
    protected = text
    abbreviations = [
        "Inc.", "Corp.", "Ltd.", "Co.", "L.P.", "LLC.", "N.A.", "S.A.",
        "Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "Prof.",
        "No.", "Nos.", "Vol.", "vs.", "etc.", "approx.", "i.e.", "e.g.",
        "U.S.", "U.K.", "E.U.", "ASC.",
    ]
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", "<<DOT>>"))

    protected = re.sub(r"(\d)\.(\d)", r"\1<<DOT>>\2", protected)
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\(])', protected)
    sentences = [s.replace("<<DOT>>", ".").strip() for s in raw]

    filtered = []
    for s in sentences:
        if len(s) < 25:
            continue
        alpha = sum(1 for c in s if c.isalpha())
        if alpha / max(len(s), 1) < 0.40:
            continue
        filtered.append(s)

    return filtered


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def extract_filing_date(filename: str) -> str:
    """Extracts filing date from filename like AAON_10-K_2024-02-28_..."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else "unknown"


def build_batches(sentences: list[str]) -> list[dict]:
    """Groups sentences into batches with context overlap."""
    total = len(sentences)
    batches = []

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        ctx_start = max(0, start - CONTEXT_WINDOW)
        ctx_end = min(total, end + CONTEXT_WINDOW)

        lines = []
        expected_ids = []

        for i in range(ctx_start, ctx_end):
            if i < start or i >= end:
                lines.append(f"[CONTEXT ONLY] Sentence {i}: {sentences[i]}")
            else:
                lines.append(f"[LABEL THIS] Sentence {i}: {sentences[i]}")
                expected_ids.append(i)

        batches.append({
            "sentences_block": "\n\n".join(lines),
            "expected_ids": expected_ids,
        })

    return batches


# ---------------------------------------------------------------------------
# MODE: EXTRACT
# ---------------------------------------------------------------------------

def run_extract():
    """Walks the filings folder and produces a sentence-level CSV."""
    print("[1/2] Scanning filings...")

    all_rows = []
    companies = sorted([d for d in FILINGS_DIR.iterdir() if d.is_dir()])
    print(f"  Found {len(companies)} company folders")

    failed_extractions = []

    for comp_idx, comp_dir in enumerate(companies):
        ticker = comp_dir.name

        for filing_type in ["10-K", "10-Q"]:
            type_dir = comp_dir / filing_type
            if not type_dir.exists():
                continue

            for filepath in sorted(type_dir.iterdir()):
                if filepath.suffix.lower() not in (".html", ".htm"):
                    continue

                filing_date = extract_filing_date(filepath.name)

                raw_text = extract_text_from_html(filepath)
                if not raw_text or len(raw_text) < 500:
                    print(f"  [!] {ticker} {filing_type} {filing_date}: no text extracted")
                    failed_extractions.append((ticker, filing_type, filing_date, "no text"))
                    continue

                raw_text = clean_text(raw_text)

                for section_name in ["Risk Factors", "MD&A"]:
                    section_text = extract_section(raw_text, section_name, filing_type)
                    if not section_text:
                        print(f"  [!] {ticker} {filing_type} {filing_date}: {section_name} not found")
                        failed_extractions.append((ticker, filing_type, filing_date, section_name))
                        continue

                    sentences = split_into_sentences(section_text)
                    if not sentences:
                        continue

                    section_code = "RF" if section_name == "Risk Factors" else "MDA"

                    for i, sent in enumerate(sentences):
                        sent_id = f"{ticker}_{filing_type}_{filing_date}_{section_code}_{i:04d}"
                        all_rows.append({
                            "sentence_id":  sent_id,
                            "ticker":       ticker,
                            "filing_type":  filing_type,
                            "filing_date":  filing_date,
                            "filename":     filepath.name,
                            "section":      section_name,
                            "sentence_num": i,
                            "text":         sent,
                        })

        if (comp_idx + 1) % 25 == 0 or comp_idx == len(companies) - 1:
            print(f"  Processed {comp_idx + 1}/{len(companies)} companies "
                  f"({len(all_rows):,} sentences so far)")

    print(f"\n[2/2] Saving sentences CSV...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(SENTENCES_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Extraction Complete")
    print(f"{'='*60}")
    print(f"  Companies:       {df['ticker'].nunique()}")
    print(f"  Filings:         {df.groupby(['ticker', 'filing_type', 'filing_date']).ngroups}")
    print(f"  Total sentences: {len(df):,}")
    print(f"  By section:")
    for sec, count in df["section"].value_counts().items():
        print(f"    {sec:<20} {count:>6,}")
    if failed_extractions:
        print(f"\n  Failed extractions: {len(failed_extractions)}")
        for ticker, ft, fd, reason in failed_extractions:
            print(f"    {ticker} {ft} {fd}: {reason}")
    print(f"\n  Output: {SENTENCES_CSV}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# MODE: BATCH
# ---------------------------------------------------------------------------

def run_batch():
    """Builds JSONL files, submits batches ONE AT A TIME, polls each to
    completion before submitting the next. Auto-downloads at the end."""
    print("[1/3] Loading sentences...")
    df = pd.read_csv(SENTENCES_CSV)
    print(f"  {len(df):,} sentences from {df['ticker'].nunique()} companies")

    print("\n[2/3] Building batch requests...")
    all_requests = []
    request_idx = 0

    groups = df.groupby(["ticker", "filing_type", "filing_date", "section", "filename"])

    for (ticker, filing_type, filing_date, section, filename), group_df in groups:
        sentences = group_df["text"].tolist()
        sentence_nums = group_df["sentence_num"].tolist()

        batches = build_batches(sentences)

        for batch in batches:
            expected_ids_remapped = []
            remapped_lines = []

            for line in batch["sentences_block"].split("\n\n"):
                match = re.match(
                    r"(\[(?:CONTEXT ONLY|LABEL THIS)\]) Sentence (\d+): (.+)",
                    line, re.DOTALL
                )
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
            num_expected = len(expected_ids_remapped)

            user_prompt = f"""Company: {ticker}
Filing: {filing_type} ({filing_date}) -- Section: {section}

Classify each sentence marked [LABEL THIS] into one of the 12 risk topics or 'other'.
Sentences marked [CONTEXT ONLY] are for context only -- do NOT label them.
You MUST return exactly {num_expected} results.

---SENTENCES---
{remapped_block}
---END---"""

            custom_id = f"{ticker}__{filing_type}__{filing_date}__{section}__{request_idx:05d}"
            custom_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", custom_id)

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
            }

            all_requests.append(request)
            request_idx += 1

    print(f"  {len(all_requests):,} total batch requests")

    # --- Split into chunks ---
    chunks = []
    for i in range(0, len(all_requests), MAX_PER_BATCH):
        chunks.append(all_requests[i:i + MAX_PER_BATCH])

    print(f"  Splitting into {len(chunks)} batch(es) of up to {MAX_PER_BATCH:,} requests each")

    # --- Write JSONL files ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_paths = []
    for chunk_idx, chunk in enumerate(chunks):
        jsonl_path = OUTPUT_DIR / f"topic_batch_requests_{chunk_idx+1}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for req in chunk:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        jsonl_paths.append(jsonl_path)

    # --- Submit and wait, one at a time ---
    print(f"\n[3/3] Submitting batches sequentially...")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_ids = []

    for chunk_idx, jsonl_path in enumerate(jsonl_paths):
        chunk_size = len(chunks[chunk_idx])
        print(f"\n  {'='*50}")
        print(f"  Batch {chunk_idx+1}/{len(chunks)} ({chunk_size:,} requests)")
        print(f"  {'='*50}")

        # --- Submit with retry ---
        batch_obj = None
        for attempt in range(120):
            try:
                with open(jsonl_path, "rb") as f:
                    uploaded = client.files.create(file=f, purpose="batch")

                batch_obj = client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"DDDS topic extraction -- part {chunk_idx+1}/{len(chunks)}"},
                )
                print(f"  Submitted: {batch_obj.id}")
                break

            except Exception as e:
                if "enqueued" in str(e).lower() or "limit" in str(e).lower():
                    print(f"  Enqueue limit hit, retrying in 60s... (attempt {attempt+1})")
                    time.sleep(60)
                else:
                    raise

        if not batch_obj:
            print(f"  [!] Failed to submit batch {chunk_idx+1} after 120 attempts. Stopping.")
            break

        batch_ids.append(batch_obj.id)

        # Save incrementally
        ids_path = OUTPUT_DIR / "batch_ids.txt"
        with open(ids_path, "w") as f:
            f.write("\n".join(batch_ids))

        # --- Poll until complete ---
        while True:
            batch_obj = client.batches.retrieve(batch_obj.id)
            completed = batch_obj.request_counts.completed
            total = batch_obj.request_counts.total
            failed = batch_obj.request_counts.failed
            pct = 100 * completed / max(total, 1)

            timestamp = datetime.now().strftime("%H:%M:%S")
            bar_len = 20
            filled = int(bar_len * completed / max(total, 1))
            bar = "█" * filled + "░" * (bar_len - filled)

            status_line = (
                f"  [{timestamp}]  {bar}  {pct:5.1f}%  "
                f"({completed:,}/{total:,})"
            )
            if failed:
                status_line += f"  [{failed:,} failed]"
            status_line += f"  ({batch_obj.status})"
            print(status_line)

            if batch_obj.status == "completed":
                print(f"  Batch {chunk_idx+1} complete!")
                break

            if batch_obj.status in ("failed", "expired", "cancelled"):
                print(f"  [!] Batch {chunk_idx+1} {batch_obj.status}. Retrying submission...")
                # Retry the whole batch
                batch_obj = None
                for attempt in range(120):
                    try:
                        with open(jsonl_path, "rb") as f:
                            uploaded = client.files.create(file=f, purpose="batch")
                        batch_obj = client.batches.create(
                            input_file_id=uploaded.id,
                            endpoint="/v1/chat/completions",
                            completion_window="24h",
                            metadata={"description": f"DDDS topic extraction -- part {chunk_idx+1}/{len(chunks)} (retry)"},
                        )
                        # Update stored batch ID
                        batch_ids[-1] = batch_obj.id
                        with open(ids_path, "w") as f:
                            f.write("\n".join(batch_ids))
                        print(f"  Resubmitted: {batch_obj.id}")
                        break
                    except Exception as e:
                        if "enqueued" in str(e).lower() or "limit" in str(e).lower():
                            print(f"  Enqueue limit hit, retrying in 60s... (attempt {attempt+1})")
                            time.sleep(60)
                        else:
                            raise

                if not batch_obj:
                    print(f"  [!] Failed to resubmit batch {chunk_idx+1}. Stopping.")
                    break
                continue

            time.sleep(WATCH_INTERVAL)

    # --- All batches done, download ---
    print(f"\n\n  All {len(batch_ids)} batches complete. Downloading results...\n")
    run_download(None)


# ---------------------------------------------------------------------------
# MODE: STATUS
# ---------------------------------------------------------------------------

def run_status(batch_id: str | None):
    """Prints current batch job status."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_ids = load_batch_ids(batch_id)

    if not batch_ids:
        print("  [!] No batch IDs found. Run --mode batch first, or pass --batch-id.")
        return

    for i, bid in enumerate(batch_ids):
        batch = client.batches.retrieve(bid)
        print(f"\n  Batch {i+1}/{len(batch_ids)}")
        print(f"  Batch ID:    {bid}")
        print(f"  Status:      {batch.status}")
        print(f"  Requests:    {batch.request_counts.completed}/{batch.request_counts.total} completed")
        if batch.request_counts.failed:
            print(f"  Failed:      {batch.request_counts.failed}")
        if batch.completed_at:
            print(f"  Completed:   {batch.completed_at}")


# ---------------------------------------------------------------------------
# MODE: WATCH
# ---------------------------------------------------------------------------

def load_batch_ids(batch_id: str | None) -> list[str]:
    """Loads batch IDs from file or uses the provided single ID."""
    ids_path = OUTPUT_DIR / "batch_ids.txt"
    if batch_id:
        return [batch_id]
    if ids_path.exists():
        return [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    return []


def run_watch(batch_id: str | None):
    """Polls all batch statuses every WATCH_INTERVAL seconds, auto-downloads on completion."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_ids = load_batch_ids(batch_id)

    if not batch_ids:
        print("  [!] No batch IDs found. Run --mode batch first, or pass --batch-id.")
        return

    print(f"  Watching {len(batch_ids)} batch(es)")
    print(f"  Polling every {WATCH_INTERVAL}s (Ctrl+C to stop)\n")

    while True:
        all_completed = True
        any_failed = False
        timestamp = datetime.now().strftime("%H:%M:%S")

        for i, bid in enumerate(batch_ids):
            batch = client.batches.retrieve(bid)
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            failed = batch.request_counts.failed
            pct = 100 * completed / max(total, 1)

            bar_len = 20
            filled = int(bar_len * completed / max(total, 1))
            bar = "█" * filled + "░" * (bar_len - filled)

            status_line = (
                f"  [{timestamp}] Batch {i+1}: {bar}  {pct:5.1f}%  "
                f"({completed:,}/{total:,})"
            )
            if failed:
                status_line += f"  [{failed:,} failed]"
            status_line += f"  ({batch.status})"
            print(status_line)

            if batch.status in ("failed", "expired", "cancelled"):
                any_failed = True
            elif batch.status != "completed":
                all_completed = False

        print()

        if any_failed:
            print("  [!] One or more batches failed. Check OpenAI dashboard.")
            return

        if all_completed:
            print("  All batches complete! Downloading results...\n")
            run_download(None)
            return

        time.sleep(WATCH_INTERVAL)


# ---------------------------------------------------------------------------
# MODE: DOWNLOAD
# ---------------------------------------------------------------------------

def run_download(batch_id: str | None):
    """Downloads results from all batches and merges with sentence data."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_ids = load_batch_ids(batch_id)

    if not batch_ids:
        print("  [!] No batch IDs found. Run --mode batch first, or pass --batch-id.")
        return

    print(f"[1/3] Downloading results from {len(batch_ids)} batch(es)...")
    all_raw_lines = []

    for i, bid in enumerate(batch_ids):
        batch = client.batches.retrieve(bid)
        if batch.status != "completed":
            print(f"  [!] Batch {i+1} ({bid}) not complete. Status: {batch.status}")
            continue

        content = client.files.content(batch.output_file_id)
        raw_text = content.text
        lines = [l for l in raw_text.strip().split("\n") if l.strip()]
        all_raw_lines.extend(lines)
        print(f"  Batch {i+1}: {len(lines):,} responses downloaded")

        if batch.error_file_id:
            error_content = client.files.content(batch.error_file_id)
            error_path = OUTPUT_DIR / f"topic_batch_errors_{i+1}.jsonl"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(error_content.text)
            print(f"  Errors saved: {error_path}")

    # Save combined raw output
    raw_path = OUTPUT_DIR / "topic_batch_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_raw_lines))
    print(f"  Combined raw output: {raw_path} ({len(all_raw_lines):,} responses)")

    print("\n[2/3] Parsing results...")
    all_label_rows = []
    failed_requests = 0

    for line in all_raw_lines:
        obj = json.loads(line)
        custom_id = obj["custom_id"]

        if obj["response"]["status_code"] != 200:
            failed_requests += 1
            continue

        body = obj["response"]["body"]
        raw_content = body["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(raw_content)
            results = parsed.get("results", [])
        except json.JSONDecodeError:
            failed_requests += 1
            continue

        parts = custom_id.split("__")
        ticker = parts[0] if len(parts) > 0 else ""
        filing_type = parts[1] if len(parts) > 1 else ""
        filing_date = parts[2] if len(parts) > 2 else ""
        section_raw = parts[3] if len(parts) > 3 else ""

        section = section_raw.replace("_", " ")
        if "Risk" in section:
            section = "Risk Factors"
        elif "MD" in section:
            section = "MD&A"

        # Flatten any nested lists
        flat_results = []
        for item in results:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)

        for r in flat_results:
            if isinstance(r, str):
                try:
                    r = json.loads(r)
                except (json.JSONDecodeError, TypeError):
                    failed_requests += 1
                    continue
            if not isinstance(r, dict):
                failed_requests += 1
                continue

            label = r.get("label", "other")

            all_label_rows.append({
                "ticker":           ticker,
                "filing_type":      filing_type,
                "filing_date":      filing_date,
                "section":          section,
                "sentence_num":     r.get("sentence"),
                "topic_label":      label,
                "topic_confidence": r.get("confidence", "low"),
                "topic_reason":     r.get("reason", ""),
            })

    labels_df = pd.DataFrame(all_label_rows)
    print(f"  Parsed {len(labels_df):,} labelled sentences")
    if failed_requests:
        print(f"  [!] {failed_requests} requests failed")

    print("\n[3/3] Merging with sentence data...")
    sentences_df = pd.read_csv(SENTENCES_CSV)

    merged = sentences_df.merge(
        labels_df,
        on=["ticker", "filing_type", "filing_date", "section", "sentence_num"],
        how="left",
    )

    merged.to_csv(LABELLED_CSV, index=False)

    labelled = merged[merged["topic_label"].notna()]
    unlabelled = merged[merged["topic_label"].isna()]
    other_count = (labelled["topic_label"] == "other").sum()

    print(f"\n{'='*60}")
    print(f"  Topic Extraction Complete")
    print(f"{'='*60}")
    print(f"  Total sentences:    {len(merged):,}")
    print(f"  Labelled:           {len(labelled):,}")
    print(f"  Unlabelled:         {len(unlabelled):,}")
    print(f"  Other:              {other_count:,} ({100*other_count/max(len(labelled),1):.1f}%)")
    print(f"\n  Topic distribution:")
    if len(labelled) > 0:
        for topic, count in labelled["topic_label"].value_counts().items():
            pct = 100 * count / len(labelled)
            print(f"    {topic:<35} {count:>6,}  ({pct:>5.1f}%)")

    print(f"\n  Companies:  {merged['ticker'].nunique()}")
    print(f"  Output:     {LABELLED_CSV}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DDDS Risk Topic Extraction Pipeline")
    parser.add_argument("--mode",
                        choices=["extract", "batch", "status", "watch", "download"],
                        required=True,
                        help="extract: HTML->sentences | batch: submit to API | "
                             "status: check job | watch: auto-poll + download | "
                             "download: retrieve results")
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Batch job ID (required for status/watch/download)")
    args = parser.parse_args()

    print(f"\n[DDDS] Risk Topic Extraction Pipeline")
    print(f"  Model:      {MODEL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Mode:       {args.mode}")
    print()

    if args.mode == "extract":
        run_extract()

    elif args.mode == "batch":
        run_batch()

    elif args.mode == "status":
        run_status(args.batch_id)

    elif args.mode == "watch":
        run_watch(args.batch_id)

    elif args.mode == "download":
        run_download(args.batch_id)


if __name__ == "__main__":
    main()
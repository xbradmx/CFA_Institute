"""
DDDS - Historical Filing Download for FinBERT Training Data
=============================================================
Downloads older 10-K and 10-Q filings from SEC EDGAR for companies
in the DDDS universe. These filings are temporally separated from
the main scoring window (Oct 2024 – Sep 2025) to avoid data leakage
when training the FinBERT vagueness and complexity classifiers.

What this script does
---------------------
1. Reads company_filing_summary.xlsx to get CIKs for the universe
2. Randomly selects N companies (default 50, seeded for reproducibility)
3. Queries the EDGAR submissions API for each company
4. Finds one 10-K and one 10-Q filed BEFORE the cutoff date
5. Downloads the filing HTML from EDGAR
6. Extracts Risk Factors and MD&A sections
7. Splits extracted text into individual sentences
8. Outputs a single CSV ready for the GPT-4o labelling pipeline (run_2)

Temporal separation
-------------------
Main scoring window:  Oct 2024 – Sep 2025
Training cutoff:      filings dated BEFORE 2023-06-01 (default)
This gives 16+ months of separation between training and scoring data.

Output CSV columns
------------------
    company       : company name
    identifier    : ticker
    cik           : SEC CIK number
    filing_type   : 10-K | 10-Q
    filed_date    : date the filing was submitted to SEC
    section       : Risk Factors | MD&A
    sentence_id   : sequential ID within the filing section
    text          : individual sentence text

Usage
-----
    python run_0_-_training_data_download.py
    python run_0_-_training_data_download.py --n-companies 50 --cutoff 2023-06-01
    python run_0_-_training_data_download.py --use-cached
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UNIVERSE_FILE  = "company_filing_summary.xlsx"
OUTPUT_CSV     = "data/training_sentences/training_sentences.csv"
CACHE_DIR      = "data/training_sentences/cache"
FILINGS_DIR    = "data/training_sentences/filings"

N_COMPANIES    = 50
CUTOFF_DATE    = "2023-06-01"   # only filings BEFORE this date
SEED           = 42

# EDGAR API
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL    = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
EDGAR_HEADERS         = {
    "User-Agent": "DDDS-Research lancaster-university connor@lancaster.ac.uk",
    "Accept-Encoding": "gzip, deflate",
}
REQUEST_DELAY = 0.12   # SEC rate limit: 10 req/s, we target ~8/s

# Section extraction
MIN_SENTENCE_CHARS = 40    # drop sentences shorter than this
MAX_SENTENCE_CHARS = 2000  # drop sentences longer than this (likely parsing errors)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SECTION DETECTION PATTERNS (reused from run_1)
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "Risk Factors": [
        r"(?:item\s+1a[\.\s]*[\-–—]?\s*)risk\s+factors",
        r"item\s+1a\.?\s*risk\s+factors",
        r"risk\s+factors",
    ],
    "MD&A": [
        r"(?:item\s+7[\.\s]*[\-–—]?\s*)management.s\s+discussion",
        r"(?:item\s+2[\.\s]*[\-–—]?\s*)management.s\s+discussion",   # 10-Q uses Item 2
        r"management.s\s+discussion\s+and\s+analysis",
    ],
}

SECTION_END_PATTERNS = [
    r"item\s+\d+[a-z]?\.?\s+\w",
    r"^\s*(?:part|item)\s+[ivxlcdm\d]+",
    r"quantitative\s+and\s+qualitative\s+disclosures",
    r"financial\s+statements\s+and\s+supplementary",
    r"controls\s+and\s+procedures",
    r"unresolved\s+staff\s+comments",
    r"legal\s+proceedings",
    r"mine\s+safety\s+disclosures",
    r"(?:item\s+\d)\.\s",
]


# ---------------------------------------------------------------------------
# UNIVERSE LOADING
# ---------------------------------------------------------------------------

def load_universe(path: str) -> pd.DataFrame:
    """Loads the company filing summary and returns a cleaned DataFrame."""
    df = pd.read_excel(path, sheet_name="Filing Summary")
    df = df.dropna(subset=["CIK", "Ticker"])
    df["CIK"] = df["CIK"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    return df


def select_companies(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Randomly selects n companies from the universe.
    Filters out companies with no CIK. Seeds for reproducibility.
    """
    valid = df[df["CIK"].str.len() > 0].copy()
    if len(valid) < n:
        print(f"  [!] Only {len(valid)} valid companies available, using all")
        return valid

    random.seed(seed)
    indices = random.sample(range(len(valid)), n)
    selected = valid.iloc[indices].reset_index(drop=True)
    return selected


# ---------------------------------------------------------------------------
# EDGAR API
# ---------------------------------------------------------------------------

def pad_cik(cik: str) -> str:
    """Zero-pads a CIK to 10 digits as required by EDGAR API."""
    return cik.zfill(10)


def fetch_submissions(cik: str, cache_dir: str, use_cached: bool) -> dict | None:
    """
    Fetches the company submissions JSON from EDGAR.
    Caches responses to avoid repeat downloads.
    """
    cache_path = os.path.join(cache_dir, f"submissions_{cik}.json")

    if use_cached and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    url = EDGAR_SUBMISSIONS_URL.format(cik=pad_cik(cik))
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"      [!] EDGAR returned {resp.status_code} for CIK {cik}")
            return None
        data = resp.json()

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)

        return data

    except Exception as e:
        print(f"      [!] Failed to fetch submissions for CIK {cik}: {e}")
        return None


def find_filing(submissions: dict, form_type: str, cutoff: str) -> dict | None:
    """
    Finds the most recent filing of the given type filed BEFORE the cutoff.
    Returns dict with accessionNumber, primaryDocument, filingDate, or None.
    """
    recent = submissions.get("filings", {}).get("recent", {})

    forms       = recent.get("form", [])
    dates       = recent.get("filingDate", [])
    accessions  = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i in range(len(forms)):
        if forms[i] == form_type and dates[i] < cutoff:
            return {
                "form":             forms[i],
                "filingDate":       dates[i],
                "accessionNumber":  accessions[i],
                "primaryDocument":  primary_docs[i],
            }

    return None


def download_filing(cik: str, filing: dict, filings_dir: str, use_cached: bool) -> str | None:
    """
    Downloads the filing HTML from EDGAR Archives.
    Returns the local file path, or None on failure.
    """
    accession_clean = filing["accessionNumber"].replace("-", "")
    filename = f"{cik}_{filing['form']}_{filing['filingDate']}.html"
    local_path = os.path.join(filings_dir, filename)

    if use_cached and os.path.exists(local_path):
        return local_path

    url = EDGAR_ARCHIVES_URL.format(
        cik=cik,
        accession=accession_clean,
        document=filing["primaryDocument"],
    )
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"      [!] Download failed ({resp.status_code}): {url}")
            return None

        os.makedirs(filings_dir, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(resp.content)

        return local_path

    except Exception as e:
        print(f"      [!] Download error: {e}")
        return None


# ---------------------------------------------------------------------------
# SECTION EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_html(path: str) -> str:
    """Reads an HTML filing and returns cleaned plain text."""
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
    """Removes extraction artefacts."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*[-–]\s*\d+\s*[-–]\s*$", "", text)
    text = re.sub(r"(?m)^\s*[Pp]age\s+\d+\s*$", "", text)
    text = re.sub(r"[ \t]{3,}", " ", text)
    text = re.sub(r"\xa0", " ", text)
    return text.strip()


def find_section(text: str, section_name: str) -> str | None:
    """
    Finds and returns the text of a named section.
    Returns None if not found.
    """
    patterns = SECTION_PATTERNS.get(section_name, [])
    lines = text.split("\n")

    for pattern in patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                # Found section start — now find end
                end_line = len(lines)
                for j in range(i + 3, min(i + 800, len(lines))):
                    for end_pattern in SECTION_END_PATTERNS:
                        if re.search(end_pattern, lines[j], re.IGNORECASE):
                            # Make sure it's not the same heading we matched
                            if j > i + 5:
                                end_line = j
                                break
                    if end_line < len(lines):
                        break

                section_text = "\n".join(lines[i:end_line])
                if len(section_text.strip()) > 500:
                    return section_text.strip()

    return None


# ---------------------------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """
    Splits section text into individual sentences.

    Uses a regex-based approach that handles common abbreviations
    and avoids splitting on decimals, abbreviations like U.S., etc.
    """
    # Remove the section heading line
    lines = text.split("\n")
    if lines:
        first_line = lines[0].strip().lower()
        if any(re.search(p, first_line, re.IGNORECASE) for p in
               SECTION_PATTERNS.get("Risk Factors", []) +
               SECTION_PATTERNS.get("MD&A", [])):
            lines = lines[1:]

    text = " ".join(lines)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence boundary regex:
    # Split on period/question/exclamation followed by space and uppercase letter
    # But NOT after common abbreviations
    abbrev = r"(?<!\bU\.S)(?<!\bU\.K)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bNo)(?<!\bvs)(?<!\bInc)(?<!\bCorp)(?<!\bLtd)(?<!\bCo)(?<!\bEtc)(?<!\be\.g)(?<!\bi\.e)"
    sentences = re.split(
        abbrev + r'(?<=[.!?])\s+(?=[A-Z"])',
        text
    )

    # Filter by length
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) < MIN_SENTENCE_CHARS:
            continue
        if len(s) > MAX_SENTENCE_CHARS:
            continue
        # Skip table-like content (high ratio of numbers/special chars)
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.5:
            continue
        cleaned.append(s)

    return cleaned


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def process_company(ticker: str, company_name: str, cik: str,
                    cutoff: str, filings_dir: str, cache_dir: str,
                    use_cached: bool) -> list[dict]:
    """
    Processes a single company: fetches submissions, downloads filings,
    extracts sections, splits into sentences.
    Returns list of sentence dicts.
    """
    rows = []

    # Fetch submissions
    submissions = fetch_submissions(cik, cache_dir, use_cached)
    if not submissions:
        return rows

    # Process both 10-K and 10-Q
    for form_type in ["10-K", "10-Q"]:
        filing = find_filing(submissions, form_type, cutoff)
        if not filing:
            print(f"      No {form_type} found before {cutoff}")
            continue

        print(f"      {form_type} dated {filing['filingDate']}", end=" ... ")

        # Download
        local_path = download_filing(cik, filing, filings_dir, use_cached)
        if not local_path:
            print("download failed")
            continue

        # Extract text
        raw_text = extract_text_from_html(local_path)
        if not raw_text or len(raw_text) < 1000:
            print("no usable text")
            continue

        text = clean_text(raw_text)

        # Extract sections
        section_count = 0
        for section_name in ["Risk Factors", "MD&A"]:
            section_text = find_section(text, section_name)
            if not section_text:
                continue

            sentences = split_into_sentences(section_text)
            for sid, sentence in enumerate(sentences):
                rows.append({
                    "company":     company_name,
                    "identifier":  ticker,
                    "cik":         cik,
                    "filing_type": form_type,
                    "filed_date":  filing["filingDate"],
                    "section":     section_name,
                    "sentence_id": f"{ticker}_{form_type}_{section_name}_{sid:04d}",
                    "text":        sentence,
                })
            section_count += len(sentences)

        print(f"{section_count} sentences")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="DDDS - Download historical filings for FinBERT training data"
    )
    parser.add_argument("--universe",     default=UNIVERSE_FILE,
                        help="Path to company_filing_summary.xlsx")
    parser.add_argument("--output",       default=OUTPUT_CSV,
                        help="Output CSV path")
    parser.add_argument("--n-companies",  default=N_COMPANIES, type=int,
                        help="Number of companies to sample")
    parser.add_argument("--cutoff",       default=CUTOFF_DATE,
                        help="Only use filings dated BEFORE this (YYYY-MM-DD)")
    parser.add_argument("--seed",         default=SEED, type=int,
                        help="Random seed for company selection")
    parser.add_argument("--use-cached",   action="store_true",
                        help="Use cached EDGAR responses and downloaded filings")
    parser.add_argument("--cache-dir",    default=CACHE_DIR)
    parser.add_argument("--filings-dir",  default=FILINGS_DIR)
    args = parser.parse_args()

    print(f"\n[DDDS] Training Data Download")
    print(f"  Universe:     {args.universe}")
    print(f"  Companies:    {args.n_companies}")
    print(f"  Cutoff:       filings before {args.cutoff}")
    print(f"  Cached mode:  {args.use_cached}")
    print(f"  Output:       {args.output}\n")

    # --- Load universe ---
    print("[1/4] Loading company universe...")
    universe = load_universe(args.universe)
    print(f"  {len(universe)} companies in universe")

    # --- Select companies ---
    print(f"\n[2/4] Selecting {args.n_companies} companies (seed={args.seed})...")
    selected = select_companies(universe, args.n_companies, args.seed)
    print(f"  Selected {len(selected)} companies")

    # --- Process each company ---
    print(f"\n[3/4] Downloading and extracting filings...\n")
    all_rows = []
    success_10k = 0
    success_10q = 0
    failed = []

    for i, row in selected.iterrows():
        ticker  = row["Ticker"]
        name    = row["Company Name"]
        cik     = row["CIK"]

        print(f"  [{i+1}/{len(selected)}] {ticker} ({name})")

        try:
            company_rows = process_company(
                ticker, name, cik, args.cutoff,
                args.filings_dir, args.cache_dir, args.use_cached
            )
            if company_rows:
                all_rows.extend(company_rows)
                forms = set(r["filing_type"] for r in company_rows)
                if "10-K" in forms:
                    success_10k += 1
                if "10-Q" in forms:
                    success_10q += 1
            else:
                failed.append(ticker)
        except Exception as e:
            print(f"      [!] Error: {e}")
            failed.append(ticker)

    if not all_rows:
        print("\n[!] No sentences extracted. Check EDGAR connectivity and cutoff date.")
        return

    # --- Save output ---
    print(f"\n[4/4] Saving output...")
    df = pd.DataFrame(all_rows)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    df.to_csv(args.output, index=False)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Training Data Extraction Summary")
    print(f"{'='*60}")
    print(f"  Companies attempted:    {len(selected)}")
    print(f"  10-K successful:        {success_10k}")
    print(f"  10-Q successful:        {success_10q}")
    if failed:
        print(f"  Failed:                 {len(failed)}  ({', '.join(failed[:10])}{'...' if len(failed)>10 else ''})")
    print(f"\n  Total sentences:        {len(df)}")
    print(f"\n  By section:")
    for section, count in df["section"].value_counts().items():
        print(f"    {section:<20} {count:>6} sentences")
    print(f"\n  By filing type:")
    for ft, count in df["filing_type"].value_counts().items():
        print(f"    {ft:<20} {count:>6} sentences")
    print(f"\n  Unique companies:       {df['identifier'].nunique()}")
    print(f"  Date range:             {df['filed_date'].min()} to {df['filed_date'].max()}")
    print(f"\n  Mean sentences per filing:")
    per_filing = df.groupby(["identifier", "filing_type"]).size()
    print(f"    10-K: {per_filing.xs('10-K', level=1).mean():.0f} sentences" if "10-K" in per_filing.index.get_level_values(1) else "")
    print(f"    10-Q: {per_filing.xs('10-Q', level=1).mean():.0f} sentences" if "10-Q" in per_filing.index.get_level_values(1) else "")
    print(f"\n  Saved to: {args.output}")
    print(f"{'='*60}")
    print(f"\n[Done] Training sentences ready for labelling.")
    print(f"  Next: python run_2_-_vague_labelling.py  --input {args.output}")
    print(f"  Next: python run_2_-_complex_labelling.py --input {args.output}")


if __name__ == "__main__":
    main()
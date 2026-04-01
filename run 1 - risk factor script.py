




#CONNOR O'KEEFFE NOTES ---------------------------------------------------------------------------------------------------





#We dont need this script right now. Not for FinBERT. Need to label risk topics for the LLM later
#For FinBERT
# - Take Risk Factor & MD&A Sentences from run 0
# - Label them vague vs specific, and simple vs complex using openai API
# - Take a specific subset of the vague and complex sentences 
# - Once we have a 2.5K per model set, we can analyse 250 sentences per model manually
# - If agreement high enough, we can then finetune FinBERT with this data


# For the LLM
# Run analysis on all the MD&A Sections and Risk Factor sections for all companies
# We pass these to OpenAI gpt 4o-mini, and it labels each sentence as a specific risk topic
# These risk topics are collated for each company, and this is used for the Graph RAG
# When comparison is done, the Graph RAG shows, here is this companies "political risk" for 2024, and here is their "political risk" for 2025. 
#This allows the temporal analysis to be done of just the specific language around each risk, rather than on everything (which would include boiler plate noise)
#Same thing for peer analysis done here aswell. 







#END OF NOTE --------------------------------------------------------------------------------------------------------







"""
DDDS - Document Ingestion & Section Extraction Pipeline
========================================================
Reads a flat folder of SEC filings and earnings call transcripts
(PDF, HTML, TXT) and consolidates extracted MD&A, Risk Factor, and
transcript sections into a single passages CSV ready for Stage 3
labelling.

What this script does
---------------------
1. Scans the input folder for all .pdf, .html/.htm, and .txt files
2. Detects filing type (10-K, 10-Q, earnings transcript) from content
3. Extracts company name and ticker/CIK from cover page or headers
4. Isolates MD&A and Risk Factor sections from filings,
   or full transcript text from earnings call transcripts
5. Assigns each passage a risk_category based on keyword matching
6. Outputs a single consolidated CSV ready for label_vagueness.py
   and label_complexity.py

Output CSV columns
------------------
    filename      : source file
    company       : detected company name
    identifier    : detected ticker or CIK
    filing_type   : 10-K | 10-Q | earnings_transcript | unknown
    section       : MD&A | Risk Factors | Transcript
    risk_category : financing | liquidity and solvency | costs |
                    future performance | financial sustainability |
                    product quality | general
    text          : extracted passage text

Usage
-----
    python extract_sections.py --input-dir data/filings
    python extract_sections.py --input-dir data/filings --output data/passages.csv
"""

import argparse
import os
import re
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_DIR  = "data/ALL SEC FILINGS"
OUTPUT_CSV = "data/passages.csv"
MIN_CHARS   = 200     # minimum characters for a passage to be kept
MAX_CHARS   = 8000    # maximum characters per passage (long sections are split)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FILING TYPE DETECTION PATTERNS
# ---------------------------------------------------------------------------
FILING_TYPE_PATTERNS = {
    "10-K": [
        r"annual\s+report\s+(?:on\s+form\s+)?10-k",
        r"form\s+10-k",
        r"10-k\s+annual\s+report",
    ],
    "10-Q": [
        r"quarterly\s+report\s+(?:on\s+form\s+)?10-q",
        r"form\s+10-q",
        r"10-q\s+quarterly\s+report",
    ],
    "earnings_transcript": [
        r"earnings\s+(?:call|conference)\s+(?:transcript|call)",
        r"quarterly\s+earnings\s+call",
        r"q[1-4]\s+\d{4}\s+earnings",
        r"results\s+of\s+operations.*conference\s+call",
        r"operator[:.]",                 # transcripts typically start with "Operator:"
        r"good\s+(?:morning|afternoon|evening),?\s+(?:and\s+)?welcome",
    ],
}

# ---------------------------------------------------------------------------
# SECTION DETECTION PATTERNS
# Ordered from most to least specific - first match wins
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "MD&A": [
        r"item\s+7\.?\s*management.s\s+discussion\s+and\s+analysis",
        r"management.s\s+discussion\s+and\s+analysis\s+of\s+financial",
        r"management.s\s+discussion\s+and\s+analysis",
        r"md&a",
    ],
    "Risk Factors": [
        r"item\s+1a\.?\s*risk\s+factors",
        r"risk\s+factors",
        r"risks\s+related\s+to\s+our\s+business",
        r"risks\s+and\s+uncertainties",
    ],
}

# Section end markers - signals the end of a section
SECTION_END_PATTERNS = [
    r"item\s+\d+[a-z]?\.?\s+\w",          # next Item heading
    r"^\s*(?:part|item)\s+[ivxlcdm\d]+",   # Part I, Part II etc
    r"quantitative\s+and\s+qualitative\s+disclosures",
    r"financial\s+statements\s+and\s+supplementary",
    r"controls\s+and\s+procedures",
    r"unresolved\s+staff\s+comments",
]

# ---------------------------------------------------------------------------
# RISK CATEGORY KEYWORD MAPPING
# Each category maps to keywords that, if present, suggest that category
# ---------------------------------------------------------------------------
RISK_CATEGORY_KEYWORDS = {
    "financing": [
        "credit facility", "revolving credit", "debt", "borrowing", "loan",
        "notes due", "senior notes", "bond", "indenture", "covenant",
        "interest rate", "refinanc", "maturity", "credit rating", "leverage",
        "capital structure", "equity offering", "debt issuance",
    ],
    "liquidity and solvency": [
        "liquidity", "cash flow", "working capital", "solvency", "going concern",
        "cash and cash equivalents", "operating cash", "free cash flow",
        "cash position", "cash burn", "covenant compliance", "default",
        "bankruptcy", "insolvency", "chapter 11",
    ],
    "costs": [
        "cost of goods", "cost of revenue", "gross margin", "operating expense",
        "raw material", "input cost", "labour cost", "labor cost", "overhead",
        "supply chain cost", "freight", "logistics cost", "manufacturing cost",
        "commodity price", "inflation", "pricing pressure",
    ],
    "future performance": [
        "outlook", "guidance", "forecast", "expectation", "anticipate",
        "forward-looking", "projected", "pipeline", "backlog", "order book",
        "future revenue", "growth", "expansion", "market opportunity",
        "we expect", "we believe", "we anticipate",
    ],
    "financial sustainability": [
        "pension", "defined benefit", "retirement obligation", "post-retirement",
        "actuarial", "environmental liability", "remediation", "restructuring",
        "impairment", "goodwill", "long-lived asset", "write-down", "write-off",
        "contingent liability", "legal settlement", "warranty obligation",
    ],
    "product quality": [
        "product recall", "product liability", "warranty", "defect", "quality control",
        "regulatory approval", "product safety", "customer complaint",
        "product performance", "specification", "compliance", "certification",
        "iso ", "product standard",
    ],
}


# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_txt(path: str) -> str:
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def extract_text_from_html(path: str) -> str:
    try:
        from bs4 import BeautifulSoup
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    # Remove script and style elements
                    for tag in soup(["script", "style", "head"]):
                        tag.decompose()
                    return soup.get_text(separator="\n")
            except UnicodeDecodeError:
                continue
        return ""
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")


def extract_text_from_pdf(path: str) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")
    except Exception as e:
        print(f"    [!] pdfplumber failed on {path}: {e}")
        return ""


def extract_text(path: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".html", ".htm"):
        return extract_text_from_html(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        return ""


# ---------------------------------------------------------------------------
# DOCUMENT METADATA DETECTION
# ---------------------------------------------------------------------------

def detect_filing_type(text: str) -> str:
    """
    Detects filing type from the first 3000 characters of the document.
    Checks 10-K and 10-Q before transcript, as transcripts have weaker signals.
    """
    sample = text[:3000].lower()

    for filing_type, patterns in FILING_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sample, re.IGNORECASE):
                return filing_type

    return "unknown"


def detect_company_info(text: str) -> tuple[str, str]:
    """
    Attempts to extract company name and ticker/CIK from the first 3000
    characters. Returns (company_name, identifier).
    Falls back to empty strings if not found.
    """
    sample = text[:3000]

    company = ""
    identifier = ""

    # Common SEC cover page patterns for company name
    name_patterns = [
        r"(?:exact name of registrant[^:]*:\s*)([A-Z][A-Za-z0-9\s,\.&\-\']{3,60}?)(?:\n|$)",
        r"(?:registrant[^:]*:\s*)([A-Z][A-Za-z0-9\s,\.&\-\']{3,60}?)(?:\n|$)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, sample, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            break

    # Ticker patterns
    ticker_patterns = [
        r"(?:trading\s+symbol[^:]*:|ticker[^:]*:)\s*([A-Z]{1,5})\b",
        r"\(([A-Z]{1,5})\)\s*(?:on|listed|traded)",
        r"nasdaq[:\s]+([A-Z]{1,5})\b",
        r"nyse[:\s]+([A-Z]{1,5})\b",
    ]
    for pattern in ticker_patterns:
        match = re.search(pattern, sample, re.IGNORECASE)
        if match:
            identifier = match.group(1).strip().upper()
            break

    # CIK fallback
    if not identifier:
        cik_match = re.search(r"(?:cik|central\s+index\s+key)[^:]*:\s*(\d{7,10})", sample, re.IGNORECASE)
        if cik_match:
            identifier = f"CIK{cik_match.group(1).strip()}"

    return company, identifier


# ---------------------------------------------------------------------------
# SECTION EXTRACTION
# ---------------------------------------------------------------------------

def find_section_boundaries(text: str, section_name: str) -> list[tuple[int, int]]:
    """
    Finds all start and end positions of a named section in the text.
    Returns list of (start, end) character position tuples.
    """
    patterns = SECTION_PATTERNS.get(section_name, [])
    boundaries = []

    lines = text.split("\n")
    line_positions = []
    pos = 0
    for line in lines:
        line_positions.append(pos)
        pos += len(line) + 1  # +1 for newline

    for pattern in patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                start_char = line_positions[i]

                # Find end of section
                end_char = len(text)
                for j in range(i + 2, min(i + 500, len(lines))):
                    for end_pattern in SECTION_END_PATTERNS:
                        if re.search(end_pattern, lines[j], re.IGNORECASE):
                            end_char = line_positions[j]
                            break
                    if end_char < len(text):
                        break

                # Only accept sections with meaningful content
                section_text = text[start_char:end_char]
                if len(section_text.strip()) >= MIN_CHARS:
                    boundaries.append((start_char, end_char))
                break  # found with this pattern, move to next search

    # Deduplicate overlapping boundaries
    if len(boundaries) > 1:
        merged = [boundaries[0]]
        for start, end in boundaries[1:]:
            if start > merged[-1][1] + 100:
                merged.append((start, end))
        return merged

    return boundaries


def extract_section(text: str, section_name: str) -> list[str]:
    """
    Extracts all instances of a section from the text.
    Returns list of section text strings.
    """
    boundaries = find_section_boundaries(text, section_name)
    sections = []

    for start, end in boundaries:
        section_text = text[start:end].strip()
        if len(section_text) >= MIN_CHARS:
            sections.append(section_text)

    return sections


def clean_text(text: str) -> str:
    """
    Cleans extracted text: removes excessive whitespace, page numbers,
    and other extraction artefacts.
    """
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page number artefacts (e.g. "- 47 -", "Page 12")
    text = re.sub(r"(?m)^\s*[-–]\s*\d+\s*[-–]\s*$", "", text)
    text = re.sub(r"(?m)^\s*[Pp]age\s+\d+\s*$", "", text)
    # Collapse runs of spaces
    text = re.sub(r"[ \t]{3,}", " ", text)
    return text.strip()


def split_long_section(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    Splits a section that exceeds max_chars into paragraph-respecting chunks.
    Tries to split on paragraph boundaries rather than mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = re.split(r"\n\n+", text)
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current.strip())
            # If a single paragraph exceeds max_chars, split on sentences
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = f"{current} {sent}".strip()
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= MIN_CHARS]


# ---------------------------------------------------------------------------
# RISK CATEGORY CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_risk_category(text: str) -> str:
    """
    Assigns the most likely risk category based on keyword frequency.
    Returns the category with the most keyword hits, or 'general' if none.
    """
    text_lower = text.lower()
    scores = {}

    for category, keywords in RISK_CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------------------------

def process_filing(path: str) -> list[dict]:
    """
    Processes a single filing file. Returns a list of passage dicts.
    """
    filename = Path(path).name
    rows = []

    # Extract raw text
    text = extract_text(path)
    if not text or len(text.strip()) < MIN_CHARS:
        print(f"    [!] No usable text extracted from {filename}")
        return []

    text = clean_text(text)

    # Detect metadata
    filing_type = detect_filing_type(text)
    company, identifier = detect_company_info(text)

    # Extract sections based on filing type
    if filing_type == "earnings_transcript":
        # For transcripts, treat the whole document as one section
        chunks = split_long_section(text)
        for chunk in chunks:
            rows.append({
                "filename":     filename,
                "company":      company,
                "identifier":   identifier,
                "filing_type":  filing_type,
                "section":      "Transcript",
                "risk_category": classify_risk_category(chunk),
                "text":         chunk,
            })

    else:
        # For 10-K/10-Q, extract MD&A and Risk Factors separately
        for section_name in ["MD&A", "Risk Factors"]:
            sections = extract_section(text, section_name)

            if not sections:
                print(f"    [!] {section_name} not found in {filename}")
                continue

            for section_text in sections:
                chunks = split_long_section(section_text)
                for chunk in chunks:
                    rows.append({
                        "filename":      filename,
                        "company":       company,
                        "identifier":    identifier,
                        "filing_type":   filing_type,
                        "section":       section_name,
                        "risk_category": classify_risk_category(chunk),
                        "text":          chunk,
                    })

    return rows


def scan_folder(input_dir: str) -> list[str]:
    """Returns all supported files in the input directory."""
    supported = {".pdf", ".html", ".htm", ".txt"}
    files = [
        str(p) for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in supported
    ]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="DDDS Document Ingestion & Section Extraction")
    parser.add_argument("--input-dir", default=INPUT_DIR,  help="Folder containing filing documents")
    parser.add_argument("--output",    default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    print(f"\n[DDDS] Document Ingestion & Section Extraction")
    print(f"  Input folder: {args.input_dir}")
    print(f"  Output:       {args.output}\n")

    # Scan folder
    print("[1/3] Scanning input folder...")
    files = scan_folder(args.input_dir)
    if not files:
        print(f"  [!] No supported files found in '{args.input_dir}'")
        print(f"  Supported formats: .pdf, .html, .htm, .txt")
        return

    ext_counts = {}
    for f in files:
        ext = Path(f).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print(f"  Found {len(files)} files: " + ", ".join(f"{v} {k}" for k, v in ext_counts.items()))

    # Process files
    print("\n[2/3] Extracting sections...")
    all_rows = []
    filing_type_counts = {}
    failed = []

    for i, path in enumerate(files):
        filename = Path(path).name
        print(f"  [{i+1}/{len(files)}] {filename}", end=" ... ")

        try:
            rows = process_filing(path)
            if rows:
                filing_type = rows[0]["filing_type"]
                filing_type_counts[filing_type] = filing_type_counts.get(filing_type, 0) + 1
                all_rows.extend(rows)
                print(f"{len(rows)} passages extracted  ({filing_type})")
            else:
                print("0 passages extracted")
                failed.append(filename)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(filename)

    if not all_rows:
        print("\n[!] No passages extracted. Check input files and formats.")
        return

    # Build and save DataFrame
    print(f"\n[3/3] Saving output...")
    df = pd.DataFrame(all_rows)

    # Drop passages below minimum length
    before = len(df)
    df = df[df["text"].str.len() >= MIN_CHARS]
    dropped = before - len(df)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"  Extraction Summary")
    print(f"{'='*55}")
    print(f"  Files processed:    {len(files) - len(failed)}/{len(files)}")
    if failed:
        print(f"  Failed:             {len(failed)}  ({', '.join(failed[:5])}{'...' if len(failed)>5 else ''})")
    print(f"  Total passages:     {len(df)}")
    if dropped:
        print(f"  Dropped (too short): {dropped}")
    print(f"\n  Filing types detected:")
    for ft, count in sorted(filing_type_counts.items()):
        print(f"    {ft:<25} {count} files")
    print(f"\n  Section breakdown:")
    for section, count in df["section"].value_counts().items():
        print(f"    {section:<25} {count} passages")
    print(f"\n  Risk category breakdown:")
    for cat, count in df["risk_category"].value_counts().items():
        print(f"    {cat:<30} {count} passages")
    print(f"\n  Companies detected: {df['identifier'].nunique()} unique identifiers")
    print(f"\n  Saved to: {args.output}")
    print(f"{'='*55}")
    print(f"\n[Done] Ready for Stage 3 labelling.")
    print(f"  Run:  python label_vagueness.py  --input {args.output}")
    print(f"  Run:  python label_complexity.py --input {args.output}")


if __name__ == "__main__":
    main()

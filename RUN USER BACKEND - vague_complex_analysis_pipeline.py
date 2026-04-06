# =============================================================================
# DDDS - Analyst Inference Pipeline
# Disclosure Degradation Detection System | The Transparency Project
# Lancaster University | CFA AI Investment Challenge 2026
#
# Usage:
#   Single file:  python analyst_pipeline.py path/to/filing.html
#   Folder:       python analyst_pipeline.py path/to/folder/
#
# Accepts: .html, .pdf, .txt (mixed folders supported)
# Outputs: Console summary + per-document CSV in data/analyst_outputs/
# =============================================================================

# =============================================================================
# CONFIG — adjust here only, do not edit below the divider
# =============================================================================

OUTLIER_SD_THRESHOLD = 1.5       # SD above doc mean demeaned score to flag outlier
MAX_OUTLIERS         = 5         # maximum flagged sentences printed to console per doc

VAGUE_MODEL_PATH   = "outputs/vagueness_model/final_vagueness_model"
COMPLEX_MODEL_PATH = "outputs/complexity_model/Final_model_save"

OUTPUT_DIR = "data/analyst_outputs"

# Hardcoded baselines — derived from training set predictions_vagueness.csv
# and predictions_complexity.csv, grouped by (filing_type, section).
# Recompute from those CSVs if the training set changes.
VAGUE_BASELINES = {
    ("10-K", "MD&A"):          0.3430,
    ("10-K", "Risk Factors"):  0.7035,
    ("10-Q", "MD&A"):          0.3590,
    ("10-Q", "Risk Factors"):  0.5313,
}

COMPLEX_BASELINES = {
    ("10-K", "MD&A"):          0.2283,
    ("10-K", "Risk Factors"):  0.0809,
    ("10-Q", "MD&A"):          0.1193,
    ("10-Q", "Risk Factors"):  0.0719,
}

# Fallback baselines used when section cannot be detected (filing_type only)
VAGUE_FALLBACK = {
    "10-K":    0.5381,
    "10-Q":    0.4152,
    "UNKNOWN": 0.4767,   # grand mean across all
}

COMPLEX_FALLBACK = {
    "10-K":    0.1485,
    "10-Q":    0.1039,
    "UNKNOWN": 0.1313,
}

# =============================================================================
# END CONFIG
# =============================================================================

import os
import sys
import re
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import pdfplumber


# ---------------------------------------------------------------------------
# Filing type detection
# ---------------------------------------------------------------------------

def detect_filing_type(filename: str, text_snippet: str) -> str:
    """
    Returns '10-K', '10-Q', or 'UNKNOWN'.
    Checks filename first, then first 3000 characters of document text.
    """
    name_upper = filename.upper()
    if "10-K" in name_upper:
        return "10-K"
    if "10-Q" in name_upper:
        return "10-Q"
    snippet = text_snippet[:3000].upper()
    if "10-K" in snippet:
        return "10-K"
    if "10-Q" in snippet:
        return "10-Q"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_html(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def parse_pdf(filepath: str) -> str:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def parse_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_document(filepath: str) -> tuple[str, str]:
    """
    Returns (raw_text, filing_type).
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".html", ".htm"):
        text = parse_html(filepath)
    elif ext == ".pdf":
        text = parse_pdf(filepath)
    elif ext == ".txt":
        text = parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    filing_type = detect_filing_type(os.path.basename(filepath), text)
    return text, filing_type


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

SECTION_PATTERNS = {
    "MD&A": [
        r"item\s+7[^a-z]",
        r"management.{0,10}discussion",
        r"management.{0,10}analysis",
    ],
    "Risk Factors": [
        r"item\s+1a[^a-z]",
        r"risk\s+factors",
    ],
}

SECTION_END_PATTERNS = [
    r"item\s+\d+[^a-z]",
    r"part\s+[ivx]+",
    r"signatures",
    r"financial\s+statements",
    r"quantitative.*qualitative.*disclosures",
]


def extract_sections(text: str) -> dict[str, str]:
    """
    Returns dict mapping section name to extracted text.
    Falls back to full document if no sections detected.
    """
    text_lower = text.lower()
    hits = {}

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                hits[section_name] = match.start()
                break

    if not hits:
        return {"UNKNOWN": text}

    # Sort hits by position
    ordered = sorted(hits.items(), key=lambda x: x[1])
    sections = {}

    for i, (name, start) in enumerate(ordered):
        # Find end: start of next known section or a generic section boundary
        if i + 1 < len(ordered):
            end = ordered[i + 1][1]
        else:
            # Look for next item heading after this section
            search_from = start + 100
            end = len(text)
            for ep in SECTION_END_PATTERNS:
                m = re.search(ep, text_lower[search_from:])
                if m:
                    candidate = search_from + m.start()
                    if candidate < end and candidate > start + 200:
                        end = candidate
                        break

        sections[name] = text[start:end]

    return sections


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def segment_sentences(text: str) -> list[str]:
    """
    Splits text into sentences. Filters out very short or whitespace-only lines.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split on sentence-ending punctuation followed by whitespace and capital letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    sentences = []
    for s in raw:
        s = s.strip()
        word_count = len(s.split())
        if word_count >= 8:          # discard fragments
            sentences.append(s)

    return sentences


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """
    Loads both FinBERT classifiers. Called once at startup.
    Returns (vague_model, vague_tokenizer, complex_model, complex_tokenizer, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading models on {device}...")

    vague_tokenizer = AutoTokenizer.from_pretrained(VAGUE_MODEL_PATH)
    vague_model     = AutoModelForSequenceClassification.from_pretrained(VAGUE_MODEL_PATH)
    vague_model.to(device).eval()

    complex_tokenizer = AutoTokenizer.from_pretrained(COMPLEX_MODEL_PATH)
    complex_model     = AutoModelForSequenceClassification.from_pretrained(COMPLEX_MODEL_PATH)
    complex_model.to(device).eval()

    print("[INFO] Models loaded.\n")
    return vague_model, vague_tokenizer, complex_model, complex_tokenizer, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(sentences: list[str], model, tokenizer, device,
                  batch_size: int = 32) -> list[tuple[float, str]]:
    """
    Returns list of (probability_of_positive_class, label_string).
    Label index 1 = VAGUE / COMPLEX; index 0 = SPECIFIC / SIMPLE.
    """
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            prob_positive = float(p[1])
            label = model.config.id2label[int(np.argmax(p))]
            results.append((prob_positive, label))
    return results


# ---------------------------------------------------------------------------
# Demeaning
# ---------------------------------------------------------------------------

def get_baseline(filing_type: str, section: str,
                 baselines: dict, fallback: dict) -> float:
    key = (filing_type, section)
    if key in baselines:
        return baselines[key]
    ft = filing_type if filing_type in fallback else "UNKNOWN"
    return fallback[ft]


def demean_scores(rows: list[dict], filing_type: str) -> list[dict]:
    """
    Adds demeaned_vague and demeaned_complex columns to each row.
    """
    for row in rows:
        section = row["section"]
        row["demeaned_vague"]   = round(
            row["vague_prob"] - get_baseline(filing_type, section, VAGUE_BASELINES, VAGUE_FALLBACK), 4
        )
        row["demeaned_complex"] = round(
            row["complex_prob"] - get_baseline(filing_type, section, COMPLEX_BASELINES, COMPLEX_FALLBACK), 4
        )
    return rows


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def find_outliers(rows: list[dict]) -> list[dict]:
    """
    Returns up to MAX_OUTLIERS rows flagged as outliers by absolute
    combined demeaned score, filtered to those exceeding OUTLIER_SD_THRESHOLD
    standard deviations above the document mean.
    """
    if not rows:
        return []

    combined = np.array([abs(r["demeaned_vague"]) + abs(r["demeaned_complex"])
                         for r in rows])
    mean_c = combined.mean()
    std_c  = combined.std()

    if std_c == 0:
        return []

    threshold = mean_c + OUTLIER_SD_THRESHOLD * std_c
    flagged = [(rows[i], combined[i]) for i in range(len(rows))
               if combined[i] >= threshold]
    flagged.sort(key=lambda x: x[1], reverse=True)

    for row, _ in flagged:
        row["outlier"] = True

    return [r for r, _ in flagged[:MAX_OUTLIERS]]


# ---------------------------------------------------------------------------
# Document-level summary
# ---------------------------------------------------------------------------

def build_summary(rows: list[dict], filing_type: str,
                  filename: str) -> dict[str, dict]:
    """
    Returns per-section means of raw and demeaned scores.
    """
    sections = {}
    for row in rows:
        s = row["section"]
        if s not in sections:
            sections[s] = {"vague_prob": [], "complex_prob": [],
                           "demeaned_vague": [], "demeaned_complex": []}
        sections[s]["vague_prob"].append(row["vague_prob"])
        sections[s]["complex_prob"].append(row["complex_prob"])
        sections[s]["demeaned_vague"].append(row["demeaned_vague"])
        sections[s]["demeaned_complex"].append(row["demeaned_complex"])

    summary = {}
    for s, vals in sections.items():
        summary[s] = {
            "sentences":          len(vals["vague_prob"]),
            "mean_vague_raw":     round(np.mean(vals["vague_prob"]), 4),
            "mean_complex_raw":   round(np.mean(vals["complex_prob"]), 4),
            "mean_vague_demeaned":   round(np.mean(vals["demeaned_vague"]), 4),
            "mean_complex_demeaned": round(np.mean(vals["demeaned_complex"]), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_report(filename: str, filing_type: str, summary: dict,
                 outliers: list[dict]) -> None:
    divider = "=" * 72
    thin    = "-" * 72

    print(f"\n{divider}")
    print(f"  DDDS Analyst Report")
    print(f"  File        : {filename}")
    print(f"  Filing type : {filing_type}")
    print(divider)

    print(f"\n{'SECTION':<20} {'SENTENCES':>9} {'VAGUE RAW':>10} "
          f"{'VAGUE ΔBASE':>12} {'CMPLX RAW':>10} {'CMPLX ΔBASE':>12}")
    print(thin)
    for section, vals in summary.items():
        vd_str = f"{vals['mean_vague_demeaned']:+.4f}"
        cd_str = f"{vals['mean_complex_demeaned']:+.4f}"
        print(f"{section:<20} {vals['sentences']:>9} "
              f"{vals['mean_vague_raw']:>10.4f} {vd_str:>12} "
              f"{vals['mean_complex_raw']:>10.4f} {cd_str:>12}")
    print(thin)

    if outliers:
        print(f"\n  FLAGGED SENTENCES (top {len(outliers)} outliers by combined demeaned score)")
        print(thin)
        for i, row in enumerate(outliers, 1):
            print(f"\n  [{i}] Section: {row['section']}")
            print(f"      Vague  : raw={row['vague_prob']:.4f}  Δbaseline={row['demeaned_vague']:+.4f}  "
                  f"label={row['vague_label']}")
            print(f"      Complex: raw={row['complex_prob']:.4f}  Δbaseline={row['demeaned_complex']:+.4f}  "
                  f"label={row['complex_label']}")
            # Truncate very long sentences for readability
            excerpt = row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"]
            print(f"      Text   : {excerpt}")
    else:
        print("\n  No sentences exceeded the outlier threshold.")

    print(f"\n{divider}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], summary: dict, filename: str,
              filing_type: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_ddds_scores.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Document-level header block
        writer.writerow(["DOCUMENT SUMMARY"])
        writer.writerow(["File", filename])
        writer.writerow(["Filing Type", filing_type])
        writer.writerow([])
        writer.writerow(["Section", "Sentences", "Mean Vague (Raw)",
                         "Mean Vague (Δbaseline)", "Mean Complex (Raw)",
                         "Mean Complex (Δbaseline)"])
        for section, vals in summary.items():
            writer.writerow([
                section,
                vals["sentences"],
                vals["mean_vague_raw"],
                vals["mean_vague_demeaned"],
                vals["mean_complex_raw"],
                vals["mean_complex_demeaned"],
            ])

        writer.writerow([])
        writer.writerow(["SENTENCE LEVEL DETAIL"])
        writer.writerow(["sentence_id", "section", "text",
                         "vague_prob", "vague_label", "demeaned_vague",
                         "complex_prob", "complex_label", "demeaned_complex",
                         "outlier"])

        for row in rows:
            writer.writerow([
                row["sentence_id"],
                row["section"],
                row["text"],
                row["vague_prob"],
                row["vague_label"],
                row["demeaned_vague"],
                row["complex_prob"],
                row["complex_label"],
                row["demeaned_complex"],
                row.get("outlier", False),
            ])

    return out_path


# ---------------------------------------------------------------------------
# Per-document pipeline
# ---------------------------------------------------------------------------

def process_document(filepath: str, vague_model, vague_tokenizer,
                     complex_model, complex_tokenizer, device) -> None:
    filename = os.path.basename(filepath)
    print(f"[INFO] Processing: {filename}")

    # Load and parse
    try:
        text, filing_type = load_document(filepath)
    except Exception as e:
        print(f"[ERROR] Could not parse {filename}: {e}\n")
        return

    if filing_type == "UNKNOWN":
        print(f"[WARN] Could not detect filing type for {filename}. "
              f"Using grand-mean baselines.")

    # Extract sections
    sections = extract_sections(text)

    # Build sentence rows
    all_sentences = []
    for section_name, section_text in sections.items():
        sentences = segment_sentences(section_text)
        for idx, sent in enumerate(sentences):
            all_sentences.append({
                "sentence_id":    f"{os.path.splitext(filename)[0]}_{section_name}_{idx:04d}",
                "section":        section_name,
                "text":           sent,
                "vague_prob":     None,
                "vague_label":    None,
                "complex_prob":   None,
                "complex_label":  None,
                "demeaned_vague":   None,
                "demeaned_complex": None,
                "outlier":          False,
            })

    if not all_sentences:
        print(f"[WARN] No sentences extracted from {filename}. Skipping.\n")
        return

    # Run inference
    texts = [r["text"] for r in all_sentences]
    vague_preds   = run_inference(texts, vague_model,   vague_tokenizer,   device)
    complex_preds = run_inference(texts, complex_model, complex_tokenizer, device)

    for i, row in enumerate(all_sentences):
        row["vague_prob"],   row["vague_label"]   = round(vague_preds[i][0], 4),   vague_preds[i][1]
        row["complex_prob"], row["complex_label"] = round(complex_preds[i][0], 4), complex_preds[i][1]

    # Demean
    all_sentences = demean_scores(all_sentences, filing_type)

    # Outliers
    outliers = find_outliers(all_sentences)

    # Summary
    summary = build_summary(all_sentences, filing_type, filename)

    # Outputs
    print_report(filename, filing_type, summary, outliers)
    out_path = write_csv(all_sentences, summary, filepath, filing_type)
    print(f"[INFO] CSV saved: {out_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".html", ".htm", ".pdf", ".txt"}


def collect_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyst_pipeline.py <file_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    files = collect_files(input_path)
    if not files:
        print(f"[ERROR] No supported files found at: {input_path}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} file(s) to process.\n")

    # Load models once
    vague_model, vague_tokenizer, complex_model, complex_tokenizer, device = load_models()

    for filepath in files:
        process_document(filepath, vague_model, vague_tokenizer,
                         complex_model, complex_tokenizer, device)

    print("[INFO] All documents processed.")


if __name__ == "__main__":
    main()
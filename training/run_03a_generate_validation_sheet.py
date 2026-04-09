"""
DDDS - Validation Sheet Generator
===================================
Samples 300 sentences per classifier (vagueness + complexity) for human validation.
- 50/50 class split: 150 label=0 + 150 label=1 per classifier
- Only uses consensus=True rows (both GPT-4o runs agreed)
- Excludes any sentences already in the existing validation_subset.csv
- Splits 300 rows evenly across 3 annotators (100 each per classifier)
- Outputs a single Excel file with two sheets: "Vague" and "Complex"

Each annotator reviews 200 sentences total (100 vague + 100 complex).

Usage
-----
    python "run 3a - generate validation sheet.py"

Output
------
    data/validation/validation_annotation_sheet.xlsx
"""

import os
import random
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ANNOTATORS       = ["Brad", "Connor", "Ebro"]
SAMPLES_PER_CLASS = 150        # 150 label=0 + 150 label=1 = 300 per classifier
RANDOM_SEED      = 42

VAGUE_LABELLED   = "data/labelled/vagueness/labelled.csv"
COMPLEX_LABELLED = "data/labelled/complexity/labelled.csv"
VAGUE_EXISTING   = "data/labelled/vagueness/validation_subset.csv"
COMPLEX_EXISTING = "data/labelled/complexity/validation_subset.csv"
OUTPUT_DIR       = "data/validation"
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "validation_annotation_sheet.xlsx")

# Label display names per classifier
LABEL_NAMES = {
    "vagueness":  {0: "SPECIFIC", 1: "VAGUE"},
    "complexity": {0: "SIMPLE",   1: "COMPLEX"},
}
# ---------------------------------------------------------------------------


def load_existing_texts(path: str) -> set:
    """Load texts already in a validation subset so we don't re-sample them."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if "text" in df.columns:
        return set(df["text"].str.strip().tolist())
    return set()


def sample_balanced(labelled_path: str, existing_texts: set, classifier: str) -> pd.DataFrame:
    """
    Sample SAMPLES_PER_CLASS rows for each class label from labelled.csv,
    using only consensus=True rows and excluding already-seen texts.
    Returns a shuffled DataFrame of 2*SAMPLES_PER_CLASS rows.
    """
    df = pd.read_csv(labelled_path)

    # Keep only clean consensus rows
    df = df[df["consensus"] == True].copy()

    # Exclude already-sampled texts
    df = df[~df["text"].str.strip().isin(existing_texts)].copy()

    # Normalise label to int
    df["label"] = df["label"].astype(float).astype(int)

    label_names = LABEL_NAMES[classifier]
    frames = []
    for lbl, lbl_name in label_names.items():
        pool = df[df["label"] == lbl]
        n_available = len(pool)
        if n_available < SAMPLES_PER_CLASS:
            print(f"  WARNING [{classifier}] label={lbl} ({lbl_name}): only {n_available} available, "
                  f"wanted {SAMPLES_PER_CLASS}. Taking all.")
            sampled = pool
        else:
            sampled = pool.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
        frames.append(sampled)
        print(f"  [{classifier}] label={lbl} ({lbl_name}): sampled {len(sampled)}/{n_available}")

    combined = pd.concat(frames).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return combined


def assign_annotators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns annotators to rows in round-robin order so each gets an equal share.
    """
    n = len(df)
    per_annotator = n // len(ANNOTATORS)
    remainder = n % len(ANNOTATORS)

    assignments = []
    for i, annotator in enumerate(ANNOTATORS):
        count = per_annotator + (1 if i < remainder else 0)
        assignments.extend([annotator] * count)

    # Shuffle annotator assignments with a fixed seed for reproducibility
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(assignments)

    df = df.copy()
    df.insert(0, "annotator", assignments)
    return df


def build_sheet(labelled_path: str, existing_path: str, classifier: str) -> pd.DataFrame:
    """Full pipeline for one classifier: sample → assign → format output columns."""
    print(f"\n[{classifier.upper()}]")
    existing_texts = load_existing_texts(existing_path)
    print(f"  Excluding {len(existing_texts)} already-sampled texts")

    sampled = sample_balanced(labelled_path, existing_texts, classifier)
    sampled = assign_annotators(sampled)

    label_names = LABEL_NAMES[classifier]

    # Build clean output DataFrame
    out = pd.DataFrame({
        "annotator":       sampled["annotator"],
        "sentence_id":     sampled["sentence_id"],
        "company":         sampled["company"],
        "filing_type":     sampled["filing_type"],
        "section":         sampled["section"],
        "text":            sampled["text"],
        "gpt4o_label":     sampled["label"].map(lambda x: f"{x} ({label_names[x]})"),
        "gpt4o_confidence":sampled["confidence"],
        "gpt4o_reason":    sampled.apply(
                               lambda r: r["reason_1"] if pd.notna(r.get("reason_1")) else "", axis=1
                           ),
        "human_label":     "",   # annotator fills in: 0 or 1
        "human_notes":     "",   # optional notes
    })

    # Sort: annotator first, then shuffle within each annotator block
    out = out.sort_values("annotator").reset_index(drop=True)
    return out


def print_summary(df: pd.DataFrame, classifier: str):
    label_names = LABEL_NAMES[classifier]
    print(f"\n  Summary for {classifier}:")
    print(f"  Total rows: {len(df)}")
    for annotator in ANNOTATORS:
        subset = df[df["annotator"] == annotator]
        print(f"    {annotator}: {len(subset)} sentences")
    raw_labels = df["gpt4o_label"].str.extract(r"^(\d)").iloc[:, 0].astype(int)
    for lbl, name in label_names.items():
        print(f"    label {lbl} ({name}): {(raw_labels == lbl).sum()}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vague_sheet   = build_sheet(VAGUE_LABELLED,   VAGUE_EXISTING,   "vagueness")
    complex_sheet = build_sheet(COMPLEX_LABELLED, COMPLEX_EXISTING, "complexity")

    print_summary(vague_sheet,   "vagueness")
    print_summary(complex_sheet, "complexity")

    # Write to Excel with two sheets
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        vague_sheet.to_excel(writer,   sheet_name="Vague",   index=False)
        complex_sheet.to_excel(writer, sheet_name="Complex", index=False)

        # Auto-size columns for readability
        for sheet_name, df in [("Vague", vague_sheet), ("Complex", complex_sheet)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, 1):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(col)
                )
                # Cap text column width, leave others natural
                if col == "text":
                    ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = 80
                elif col in ("gpt4o_reason", "human_notes"):
                    ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = 50
                else:
                    ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 2, 40)

    print(f"\n[Done] Output written to: {OUTPUT_FILE}")
    print(f"  Each annotator fills in the 'human_label' column (0 or 1) for their assigned rows.")
    print(f"  Vague sheet:   0=SPECIFIC, 1=VAGUE")
    print(f"  Complex sheet: 0=SIMPLE,   1=COMPLEX")


if __name__ == "__main__":
    main()

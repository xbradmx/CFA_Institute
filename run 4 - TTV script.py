"""
DDDS - Training Data Preparation
=================================
Bridges Stage 3 (labelling) and Stage 1 (fine-tuning) by taking the
labelled.csv output from either label_vagueness.py or label_complexity.py,
selecting the required columns, and splitting into train/val/test CSVs
ready for the corresponding fine-tuning pipeline.

Keeps vagueness and complexity splits entirely separate so each
classifier trains on its own independently validated dataset.

Usage
-----
    python prepare_training_data.py --classifier vagueness \\
        --input  data/labelled/vagueness/labelled.csv \\
        --output-dir data/training/vagueness

    python prepare_training_data.py --classifier complexity \\
        --input  data/labelled/complexity/labelled.csv \\
        --output-dir data/training/complexity
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VAGUENESS_INPUT      = "data/labelled/vagueness/labelled.csv"
VAGUENESS_OUTPUT_DIR = "data/training/vagueness"

COMPLEXITY_INPUT      = "data/labelled/complexity/labelled.csv"
COMPLEXITY_OUTPUT_DIR = "data/training/complexity"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15     # must equal 1 - TRAIN_FRAC - VAL_FRAC

SEED = 42
# ---------------------------------------------------------------------------

CLASSIFIER_CONFIG = {
    "vagueness": {
        "input":      VAGUENESS_INPUT,
        "output_dir": VAGUENESS_OUTPUT_DIR,
        "label_0":    "SPECIFIC",
        "label_1":    "VAGUE",
    },
    "complexity": {
        "input":      COMPLEXITY_INPUT,
        "output_dir": COMPLEXITY_OUTPUT_DIR,
        "label_0":    "SIMPLE",
        "label_1":    "COMPLEX",
    },
}


def load_and_validate(path: str, classifier: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Required columns from labelled.csv
    required = {"text", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Make sure you are pointing at labelled.csv from label_{classifier}.py"
        )

    # Select only the columns needed for fine-tuning
    # Keep risk_category if present for traceability, but don't require it
    cols = ["text", "label"]
    if "risk_category" in df.columns:
        cols.append("risk_category")
    df = df[cols].copy()

    # Coerce and validate labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid):
        raise ValueError(
            f"label column contains values other than 0 or 1 after coercion.\n"
            f"Found: {invalid['label'].unique()}\n"
            f"This may indicate None values from failed consensus rows. "
            f"Check needs_review.csv and remove or resolve these rows."
        )

    df["label"] = df["label"].astype(int)

    # Drop missing text
    before = len(df)
    df = df.dropna(subset=["text"])
    dropped = before - len(df)
    if dropped:
        print(f"  [!] Dropped {dropped} rows with missing text")

    df["text"] = df["text"].astype(str).str.strip()
    df = df.reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split.
    Stratification preserves the label distribution across all three splits
    so class balance is consistent between train, val, and test.
    """
    # First cut: split off 30% as temp (will become val + test)
    train, temp = train_test_split(
        df,
        test_size=(1 - TRAIN_FRAC),
        stratify=df["label"],
        random_state=SEED,
    )

    # Second cut: split temp equally into val and test (50/50 of the 30%)
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=SEED,
    )

    return train, val, test


def print_split_summary(train, val, test, label_0: str, label_1: str):
    total = len(train) + len(val) + len(test)

    print(f"\n  {'Split':<8}  {'Total':>6}  {label_0:>10}  {label_1:>10}  {'% of data':>10}")
    print(f"  {'-'*50}")

    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        n       = len(df)
        n0      = (df["label"] == 0).sum()
        n1      = (df["label"] == 1).sum()
        pct     = 100 * n / total
        print(f"  {name:<8}  {n:>6}  {n0:>10}  {n1:>10}  {pct:>9.1f}%")

    print(f"  {'-'*50}")
    print(f"  {'Total':<8}  {total:>6}")


def main():
    parser = argparse.ArgumentParser(description="DDDS Training Data Preparation")
    parser.add_argument(
        "--classifier",
        required=True,
        choices=["vagueness", "complexity"],
        help="Which classifier to prepare data for"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to labelled.csv (overrides default for chosen classifier)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write train/val/test CSVs (overrides default)"
    )
    args = parser.parse_args()

    config     = CLASSIFIER_CONFIG[args.classifier]
    input_path = args.input      or config["input"]
    output_dir = args.output_dir or config["output_dir"]
    label_0    = config["label_0"]
    label_1    = config["label_1"]

    print(f"\n[DDDS] Training Data Preparation  --  {args.classifier.capitalize()} Classifier")
    print(f"  Input:      {input_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Split:      {int(TRAIN_FRAC*100)}/{int(VAL_FRAC*100)}/{int(TEST_FRAC*100)}  "
          f"(train/val/test)\n")

    # --- Load ---
    print("[1/3] Loading and validating labelled data...")
    df = load_and_validate(input_path, args.classifier)
    print(f"  {len(df)} passages loaded")
    print(f"  {label_1}: {(df['label']==1).sum()}  |  "
          f"{label_0}: {(df['label']==0).sum()}")

    # Warn if dataset is very small
    if len(df) < 100:
        print(f"\n  [!] Warning: only {len(df)} passages. "
              f"Fine-tuning on fewer than 100 samples may produce unreliable results.")

    # --- Split ---
    print("\n[2/3] Splitting into train / val / test...")
    train, val, test = split_data(df)
    print_split_summary(train, val, test, label_0, label_1)

    # --- Save ---
    print(f"\n[3/3] Saving splits to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path,     index=False)
    test.to_csv(test_path,   index=False)

    print(f"  train.csv  ->  {train_path}")
    print(f"  val.csv    ->  {val_path}")
    print(f"  test.csv   ->  {test_path}")

    print(f"\n[Done] Data ready for fine-tuning.")
    print(f"\n  Run:")
    classifier_dir = "vagueness_classifier" if args.classifier == "vagueness" else "complexity_classifier"
    print(f"    cd ../{classifier_dir}")
    print(f"    python train.py \\")
    print(f"        --train {train_path} \\")
    print(f"        --val   {val_path}   \\")
    print(f"        --test  {test_path}")


if __name__ == "__main__":
    main()

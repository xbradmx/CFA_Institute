"""
prepare_training_data.py
========================
Prepares the final 3,000-sentence training dataset for each FinBERT classifier
and splits it 70/15/15 into train/val/test.

Sampling strategy
-----------------
All 300 human-validated sentences are always included.
The remaining 2,700 are sampled from the GPT-labelled pool, stratified by label
to preserve class balance.

Split strategy
--------------
The 70/15/15 split is stratified on a combined key (label, human_validated),
guaranteeing that both class balance and human-validated representation are
preserved across train, val, and test.

Expected split composition (approximate)
-----------------------------------------
    Train (70%, 2,100 rows):  ~210 human-validated + ~1,890 GPT-labelled
    Val   (15%,   450 rows):  ~ 45 human-validated +   ~405 GPT-labelled
    Test  (15%,   450 rows):  ~ 45 human-validated +   ~405 GPT-labelled

Usage
-----
    python prepare_training_data.py --classifier vagueness
    python prepare_training_data.py --classifier complexity
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

TOTAL_SAMPLE = 3000   # total sentences per classifier
HUMAN_N      = 300    # human-validated rows — always all included
GPT_N        = TOTAL_SAMPLE - HUMAN_N   # 2,700 sampled from GPT pool

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC is implicitly 0.15

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

    required = {"text", "label", "human_validated"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"labelled.csv is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Run merge_human_labels.py before this script — it adds the "
            f"'human_validated' column needed for stratified sampling."
        )

    cols = ["text", "label", "human_validated"]
    if "sentence_id" in df.columns:
        cols.append("sentence_id")
    if "risk_category" in df.columns:
        cols.append("risk_category")
    df = df[cols].copy()

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid):
        raise ValueError(
            f"label column contains values other than 0 or 1.\n"
            f"Found: {invalid['label'].unique()}\n"
            f"Check needs_review.csv and resolve any failed consensus rows."
        )
    df["label"]           = df["label"].astype(int)
    df["human_validated"] = df["human_validated"].astype(int)

    before = len(df)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  [!] Dropped {dropped} rows with missing text")

    df["text"] = df["text"].astype(str).str.strip()
    return df


def sample_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Include all human-validated rows.
    Sample GPT_N rows from the remaining GPT-labelled pool, stratified by label.
    """
    human = df[df["human_validated"] == 1].copy()
    gpt   = df[df["human_validated"] == 0].copy()

    actual_human = len(human)
    if actual_human != HUMAN_N:
        print(f"  [!] Expected {HUMAN_N} human-validated rows, found {actual_human}. "
              f"Proceeding with all {actual_human}.")

    if len(gpt) < GPT_N:
        raise ValueError(
            f"GPT-labelled pool has only {len(gpt)} rows but {GPT_N} are needed. "
            f"Reduce TOTAL_SAMPLE or increase the labelled dataset."
        )

    gpt_sample, _ = train_test_split(
        gpt,
        train_size=GPT_N,
        stratify=gpt["label"],
        random_state=SEED,
    )

    combined = pd.concat([human, gpt_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Human-validated rows included:  {actual_human}")
    print(f"  GPT-labelled rows sampled:      {len(gpt_sample)}")
    print(f"  Total dataset size:             {len(combined)}")
    return combined


def split_dataset(df: pd.DataFrame):
    """
    Stratified 70/15/15 split on combined key (label, human_validated).
    This guarantees both class balance and human-validated representation
    are preserved across all three splits.
    """
    strat_key = df["label"].astype(str) + "_" + df["human_validated"].astype(str)

    train, temp = train_test_split(
        df,
        test_size=(1 - TRAIN_FRAC),
        stratify=strat_key,
        random_state=SEED,
    )

    strat_key_temp = temp["label"].astype(str) + "_" + temp["human_validated"].astype(str)

    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=strat_key_temp,
        random_state=SEED,
    )

    return train, val, test


def print_summary(train, val, test, label_0, label_1):
    total = len(train) + len(val) + len(test)
    print(f"\n  {'Split':<8}  {'N':>5}  {'%':>5}  "
          f"{'Human':>6}  {'GPT':>6}  {label_0:>10}  {label_1:>10}")
    print(f"  {'-'*60}")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        n      = len(df)
        pct    = 100 * n / total
        human  = df["human_validated"].sum()
        gpt    = n - human
        n0     = (df["label"] == 0).sum()
        n1     = (df["label"] == 1).sum()
        print(f"  {name:<8}  {n:>5}  {pct:>4.1f}%  "
              f"{human:>6}  {gpt:>6}  {n0:>10}  {n1:>10}")
    print(f"  {'-'*60}")
    print(f"  {'Total':<8}  {total:>5}")


def main():
    parser = argparse.ArgumentParser(description="DDDS Training Data Preparation")
    parser.add_argument(
        "--classifier",
        required=True,
        choices=["vagueness", "complexity"],
        help="Which classifier to prepare data for"
    )
    parser.add_argument("--input",      default=None, help="Override input labelled.csv path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config     = CLASSIFIER_CONFIG[args.classifier]
    input_path = args.input      or config["input"]
    output_dir = args.output_dir or config["output_dir"]
    label_0    = config["label_0"]
    label_1    = config["label_1"]

    print(f"\n[DDDS] Training Data Preparation — {args.classifier.capitalize()} Classifier")
    print(f"  Input:       {input_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Total rows:  {TOTAL_SAMPLE}  ({HUMAN_N} human + {GPT_N} GPT-sampled)")
    print(f"  Split:       70 / 15 / 15  (train / val / test)")
    print(f"  Stratify on: label + human_validated\n")

    print("[1/3] Loading and validating labelled data...")
    df = load_and_validate(input_path, args.classifier)
    print(f"  Total rows in labelled.csv:     {len(df)}")
    print(f"  Human-validated:                {df['human_validated'].sum()}")
    print(f"  GPT-labelled:                   {(df['human_validated'] == 0).sum()}")
    print(f"  {label_1}:  {(df['label'] == 1).sum()}  |  {label_0}: {(df['label'] == 0).sum()}")

    print("\n[2/3] Sampling 3,000-sentence dataset...")
    sampled = sample_dataset(df)

    print("\n[3/3] Splitting into train / val / test...")
    train, val, test = split_dataset(sampled)
    print_summary(train, val, test, label_0, label_1)

    print(f"\n[4/4] Saving to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path,     index=False)
    test.to_csv(test_path,   index=False)

    print(f"  train.csv → {train_path}")
    print(f"  val.csv   → {val_path}")
    print(f"  test.csv  → {test_path}")
    print(f"\n✓ Done. Ready for Run 5 (FinBERT fine-tuning).")


if __name__ == "__main__":
    main()
"""
DDDS - Inter-Annotator Agreement Checker
=========================================
Runs after all three team members have independently filled in the
'human_label' column in validation_subset.csv.

What this script does
---------------------
1. Loads the three completed validation CSVs (one per annotator)
2. Calculates inter-annotator agreement among the human reviewers
3. Calculates GPT-4o vs human agreement against the majority human label
4. Reports whether the 90% agreement threshold has been met
5. Exports a disagreement report for iterative refinement

If the 90% threshold is NOT met, the script prints guidance on which
passages drove disagreements so you can refine the labelling definitions
and re-run the labelling pipeline.

Input
-----
Three CSV files, one per annotator, each being a completed copy of
validation_subset.csv with the 'human_label' column filled in (0 or 1).

Usage
-----
    python check_agreement.py \\
        --annotator1 data/validation/brad.csv \\
        --annotator2 data/validation/connor.csv \\
        --annotator3 data/validation/ebro.csv \\
        --classifier vagueness

    # classifier is either 'vagueness' or 'complexity'
    # used only for display labels in the report
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
AGREEMENT_THRESHOLD = 0.90    # 90% required per proposal spec
OUTPUT_DIR          = "data/validation"
# ---------------------------------------------------------------------------


def load_annotator_csv(path: str, annotator_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"text", "gpt4o_label", "human_label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV for {annotator_name} is missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Check human_label is filled in
    blank = df["human_label"].isna() | (df["human_label"].astype(str).str.strip() == "")
    if blank.any():
        raise ValueError(
            f"{annotator_name}: {blank.sum()} rows have blank 'human_label'. "
            f"All rows must be labelled before running this script."
        )

    df["human_label"] = df["human_label"].astype(int)

    invalid = df[~df["human_label"].isin([0, 1])]
    if len(invalid):
        raise ValueError(
            f"{annotator_name}: human_label must be 0 or 1. "
            f"Found: {invalid['human_label'].unique()}"
        )

    return df


def compute_pairwise_agreement(labels_a: pd.Series, labels_b: pd.Series, name_a: str, name_b: str) -> dict:
    """
    Computes raw agreement percentage and Cohen's Kappa between two annotators.
    """
    agreed    = (labels_a == labels_b).sum()
    total     = len(labels_a)
    pct       = agreed / total
    kappa     = cohen_kappa_score(labels_a, labels_b)

    return {
        "pair":      f"{name_a} vs {name_b}",
        "agreed":    agreed,
        "total":     total,
        "pct":       pct,
        "kappa":     kappa,
        "passed":    pct >= AGREEMENT_THRESHOLD,
    }


def majority_label(row: pd.Series) -> int:
    """Returns the majority vote across three annotators (2 out of 3)."""
    votes = [row["label_1"], row["label_2"], row["label_3"]]
    return int(round(sum(votes) / len(votes)))


def compute_gpt4o_agreement(merged: pd.DataFrame) -> dict:
    """
    Compares GPT-4o labels against the majority human label.
    """
    gpt4o_labels  = merged["gpt4o_label"].astype(int)
    human_majority = merged["majority_label"].astype(int)

    agreed = (gpt4o_labels == human_majority).sum()
    total  = len(merged)
    pct    = agreed / total
    kappa  = cohen_kappa_score(gpt4o_labels, human_majority)

    return {
        "pair":   "GPT-4o vs Human Majority",
        "agreed": agreed,
        "total":  total,
        "pct":    pct,
        "kappa":  kappa,
        "passed": pct >= AGREEMENT_THRESHOLD,
    }


def build_disagreement_report(merged: pd.DataFrame, classifier: str) -> pd.DataFrame:
    """
    Identifies passages where GPT-4o disagreed with the human majority,
    or where humans disagreed with each other.
    """
    label_0, label_1 = ("SPECIFIC", "VAGUE") if classifier == "vagueness" else ("SIMPLE", "COMPLEX")

    rows = []
    for _, row in merged.iterrows():
        human_votes   = [row["label_1"], row["label_2"], row["label_3"]]
        human_unanimous = len(set(human_votes)) == 1
        gpt4o_agrees  = row["gpt4o_label"] == row["majority_label"]

        if not human_unanimous or not gpt4o_agrees:
            rows.append({
                "text":               row["text"],
                "gpt4o_label":        f"{row['gpt4o_label']} ({label_0 if row['gpt4o_label']==0 else label_1})",
                "annotator_1_label":  f"{row['label_1']} ({label_0 if row['label_1']==0 else label_1})",
                "annotator_2_label":  f"{row['label_2']} ({label_0 if row['label_2']==0 else label_1})",
                "annotator_3_label":  f"{row['label_3']} ({label_0 if row['label_3']==0 else label_1})",
                "majority_label":     f"{row['majority_label']} ({label_0 if row['majority_label']==0 else label_1})",
                "human_unanimous":    human_unanimous,
                "gpt4o_agrees":       gpt4o_agrees,
                "gpt4o_reason":       row.get("gpt4o_reason", ""),
            })

    return pd.DataFrame(rows)


def print_results(results: list[dict], gpt4o_result: dict, classifier: str):
    label_0, label_1 = ("SPECIFIC", "VAGUE") if classifier == "vagueness" else ("SIMPLE", "COMPLEX")

    print(f"\n{'='*60}")
    print(f"  Agreement Report  --  {classifier.capitalize()} Classifier")
    print(f"  Labels: 0={label_0}  1={label_1}")
    print(f"  Threshold: {AGREEMENT_THRESHOLD*100:.0f}%")
    print(f"{'='*60}\n")

    print("  Human Inter-Annotator Agreement:")
    all_human_passed = True
    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"    {r['pair']:<30}  {r['pct']*100:.1f}%  kappa={r['kappa']:.3f}  [{status}]")
        if not r["passed"]:
            all_human_passed = False

    print(f"\n  GPT-4o vs Human Majority:")
    status = "PASS ✓" if gpt4o_result["passed"] else "FAIL ✗"
    print(f"    {gpt4o_result['pair']:<30}  {gpt4o_result['pct']*100:.1f}%  "
          f"kappa={gpt4o_result['kappa']:.3f}  [{status}]")

    overall = all_human_passed and gpt4o_result["passed"]
    print(f"\n{'='*60}")
    if overall:
        print(f"  OVERALL: THRESHOLD MET ✓")
        print(f"  GPT-4o labelling is validated. Proceed to fine-tuning.")
    else:
        print(f"  OVERALL: THRESHOLD NOT MET ✗")
        print(f"  Review disagreement_report.csv and refine:")
        print(f"    1. Update the definitions and boundary examples in the labelling prompt")
        print(f"    2. Re-run label_{classifier}.py on the full dataset")
        print(f"    3. Re-run this script with fresh annotations")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="DDDS Inter-Annotator Agreement Checker")
    parser.add_argument("--annotator1",  required=True, help="CSV from annotator 1")
    parser.add_argument("--annotator2",  required=True, help="CSV from annotator 2")
    parser.add_argument("--annotator3",  required=True, help="CSV from annotator 3")
    parser.add_argument("--classifier",  required=True, choices=["vagueness", "complexity"],
                        help="Which classifier this validation is for")
    parser.add_argument("--output-dir",  default=OUTPUT_DIR, help="Directory for output reports")
    args = parser.parse_args()

    print(f"\n[DDDS] Inter-Annotator Agreement Check  --  {args.classifier.capitalize()}")

    # --- Load ---
    print("\n[1/4] Loading annotator CSVs...")
    df1 = load_annotator_csv(args.annotator1, "Annotator 1")
    df2 = load_annotator_csv(args.annotator2, "Annotator 2")
    df3 = load_annotator_csv(args.annotator3, "Annotator 3")

    # Align on text as key (all three must have the same rows)
    if not (len(df1) == len(df2) == len(df3)):
        raise ValueError(
            f"Annotator CSVs have different row counts: "
            f"{len(df1)}, {len(df2)}, {len(df3)}. They must all be the same file."
        )

    print(f"  {len(df1)} passages per annotator\n")

    # --- Merge ---
    print("[2/4] Merging and computing majority labels...")
    merged = df1[["text", "gpt4o_label", "gpt4o_reason"]].copy()
    merged["label_1"] = df1["human_label"].values
    merged["label_2"] = df2["human_label"].values
    merged["label_3"] = df3["human_label"].values
    merged["majority_label"] = merged.apply(majority_label, axis=1)

    # --- Pairwise human agreement ---
    print("[3/4] Computing agreement metrics...")
    pairwise = [
        compute_pairwise_agreement(merged["label_1"], merged["label_2"], "Annotator 1", "Annotator 2"),
        compute_pairwise_agreement(merged["label_1"], merged["label_3"], "Annotator 1", "Annotator 3"),
        compute_pairwise_agreement(merged["label_2"], merged["label_3"], "Annotator 2", "Annotator 3"),
    ]
    gpt4o_result = compute_gpt4o_agreement(merged)

    print_results(pairwise, gpt4o_result, args.classifier)

    # --- Disagreement report ---
    print("[4/4] Exporting disagreement report...")
    os.makedirs(args.output_dir, exist_ok=True)
    disagree_df = build_disagreement_report(merged, args.classifier)

    if len(disagree_df):
        report_path = os.path.join(args.output_dir, f"disagreement_report_{args.classifier}.csv")
        disagree_df.to_csv(report_path, index=False)
        print(f"  {len(disagree_df)} disagreements written to: {report_path}")
    else:
        print(f"  No disagreements found.")

    print("\n[Done]")


if __name__ == "__main__":
    main()

"""
merge_human_labels.py
=====================
Merges completed human annotations from the validation annotation sheet
into labelled.csv for both classifiers.

For the 300 human-validated sentences, the human label replaces the GPT-4o
label and a 'human_validated' flag is set to 1.
For all remaining sentences, the GPT-4o label is kept and 'human_validated'
is set to 0.

The 'human_validated' column is used downstream by prepare_training_data.py
to guarantee human-validated rows are present in all three splits.

Usage
-----
    python merge_human_labels.py

Expects
-------
    validation_annotation_sheet.xlsx   (the completed annotation sheet)
    data/labelled/vagueness/labelled.csv
    data/labelled/complexity/labelled.csv

Outputs
-------
    data/labelled/vagueness/labelled.csv   (updated in place, original backed up)
    data/labelled/complexity/labelled.csv  (updated in place, original backed up)
"""

import os
import shutil
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — update XLSX_PATH if your file is named differently
# ---------------------------------------------------------------------------
XLSX_PATH = "validation_annotation_sheet.xlsx"

VAGUENESS_CSV  = "data/labelled/vagueness/labelled.csv"
COMPLEXITY_CSV = "data/labelled/complexity/labelled.csv"
# ---------------------------------------------------------------------------


def backup(path: str) -> str:
    backup_path = path.replace(".csv", "_gpt_labels_backup.csv")
    shutil.copy2(path, backup_path)
    return backup_path


def merge(labelled_path: str, xlsx_path: str, sheet_name: str, sheet_kwargs: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {sheet_name}")
    print(f"{'='*60}")

    # Load completed annotation sheet
    annot = pd.read_excel(xlsx_path, sheet_name=sheet_name, **sheet_kwargs)
    annot = annot.dropna(subset=["human_label"])
    annot["human_label"] = annot["human_label"].astype(int)
    human_map = dict(zip(annot["sentence_id"], annot["human_label"]))
    print(f"  Human labels loaded:      {len(human_map)} sentences")

    # Load labelled.csv
    df = pd.read_csv(labelled_path)
    print(f"  labelled.csv rows:        {len(df)}")

    if "sentence_id" not in df.columns:
        raise ValueError(
            f"labelled.csv does not have a 'sentence_id' column.\n"
            f"Found columns: {df.columns.tolist()}\n"
            f"Check your Run 2 labelling script output."
        )

    # Back up before modifying
    backup_path = backup(labelled_path)
    print(f"  Backup saved to:          {backup_path}")

    # Apply human labels and stamp human_validated flag
    before = df["label"].copy()

    df["label"] = df.apply(
        lambda row: human_map[row["sentence_id"]]
        if row["sentence_id"] in human_map
        else row["label"],
        axis=1
    )

    df["human_validated"] = df["sentence_id"].isin(human_map).astype(int)

    overrides = (df["label"] != before).sum()
    matched   = df["human_validated"].sum()
    unchanged = matched - overrides

    print(f"  Rows matched to human annotations: {matched}")
    print(f"  Labels overridden (human != GPT):  {overrides}")
    print(f"  Labels unchanged (human == GPT):   {unchanged}")
    print(f"  Rows kept with GPT label:          {len(df) - matched}")
    print(f"  'human_validated' column added:    1={matched}, 0={len(df) - matched}")

    df.to_csv(labelled_path, index=False)
    print(f"  Saved: {labelled_path}")


def main():
    if not os.path.exists(XLSX_PATH):
        raise FileNotFoundError(
            f"Could not find annotation sheet at: {XLSX_PATH}\n"
            f"Update XLSX_PATH at the top of this script to match your filename."
        )

    for path in [VAGUENESS_CSV, COMPLEXITY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find: {path}\n"
                f"Run 2 (GPT-4o labelling) must be complete before running this script."
            )

    merge(
        labelled_path=VAGUENESS_CSV,
        xlsx_path=XLSX_PATH,
        sheet_name="Vague",
        sheet_kwargs={}
    )

    merge(
        labelled_path=COMPLEXITY_CSV,
        xlsx_path=XLSX_PATH,
        sheet_name="Complex",
        sheet_kwargs={"header": 1}
    )

    print("\n✓ Both classifiers updated. Ready for prepare_training_data.py.")


if __name__ == "__main__":
    main()
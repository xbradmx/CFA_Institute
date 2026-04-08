"""
Creates a small test subset of sentences_extracted.csv for batch API validation.
Backs up the real CSV, swaps in the test subset, then you run --mode batch/download.
After testing, run with --restore to put the real CSV back.

Usage
-----
    python test_batch_subset.py                  # Create test subset
    python test_batch_subset.py --restore        # Restore full CSV after testing
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

SENTENCES_CSV = Path("data/labelled/topics/sentences_extracted.csv")
BACKUP_CSV    = Path("data/labelled/topics/sentences_extracted_FULL.csv")
TEST_COMPANIES = ["AAON", "AAPL", "ACA"]


def create_subset():
    if BACKUP_CSV.exists():
        print(f"[!] Backup already exists at {BACKUP_CSV}")
        print(f"    Run with --restore first, or delete it manually.")
        return

    df = pd.read_csv(SENTENCES_CSV)
    print(f"Full CSV: {len(df):,} sentences, {df['ticker'].nunique()} companies")

    test_df = df[df["ticker"].isin(TEST_COMPANIES)]
    print(f"Test CSV: {len(test_df):,} sentences, {test_df['ticker'].nunique()} companies")
    print(f"  Companies: {', '.join(test_df['ticker'].unique())}")
    print(f"  Filings:   {test_df.groupby(['ticker', 'filing_type', 'filing_date']).ngroups}")

    # Backup full, replace with test
    shutil.copy2(SENTENCES_CSV, BACKUP_CSV)
    test_df.to_csv(SENTENCES_CSV, index=False)

    print(f"\n  Full CSV backed up to: {BACKUP_CSV}")
    print(f"  Test CSV written to:   {SENTENCES_CSV}")
    print(f"\n  Now run: python \"run 1 - Topic Extraction.py\" --mode batch")


def restore():
    if not BACKUP_CSV.exists():
        print("[!] No backup found — nothing to restore.")
        return

    shutil.copy2(BACKUP_CSV, SENTENCES_CSV)
    BACKUP_CSV.unlink()
    print(f"Restored full CSV and removed backup.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore()
    else:
        create_subset()


if __name__ == "__main__":
    main()
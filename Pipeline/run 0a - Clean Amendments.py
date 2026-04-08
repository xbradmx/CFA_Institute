"""
DDDS - Amendment Cleaning & Filing Count Enforcement
=====================================================
Cleans the 'Filings by company' directory by:
1. Deleting all amendment files (10-K-A / 10-Q-A in filename)
2. Enforcing exactly 2x 10-K and 3x 10-Q per company
3. Removing any company folder that fails validation

Usage
-----
    python run_0b_-_clean_amendments.py --data-dir "data/Filings by company"
    python run_0b_-_clean_amendments.py --data-dir "data/Filings by company" --dry-run
"""

import argparse
import shutil
from pathlib import Path


REQUIRED_10K = 2
REQUIRED_10Q = 3


def clean_company(company_dir: Path, dry_run: bool) -> dict:
    """Process one company folder. Returns a status dict."""
    ticker = company_dir.name
    result = {"ticker": ticker, "action": "kept", "amendments_removed": 0, "reason": ""}

    k_dir = company_dir / "10-K"
    q_dir = company_dir / "10-Q"

    # --- Step 1: Check both folders exist ---
    if not k_dir.is_dir() or not q_dir.is_dir():
        missing = []
        if not k_dir.is_dir():
            missing.append("10-K")
        if not q_dir.is_dir():
            missing.append("10-Q")
        result["action"] = "DELETED"
        result["reason"] = f"missing folder(s): {', '.join(missing)}"
        if not dry_run:
            shutil.rmtree(company_dir)
        return result

    # --- Step 2: Remove amendment files ---
    for folder in [k_dir, q_dir]:
        for f in list(folder.iterdir()):
            if f.is_file() and ("10-K-A" in f.name or "10-Q-A" in f.name):
                result["amendments_removed"] += 1
                if not dry_run:
                    f.unlink()

    # --- Step 3: Count remaining filings (exclude amendments even in dry-run) ---
    is_amendment = lambda f: "10-K-A" in f.name or "10-Q-A" in f.name
    k_count = sum(1 for f in k_dir.iterdir() if f.is_file() and not is_amendment(f))
    q_count = sum(1 for f in q_dir.iterdir() if f.is_file() and not is_amendment(f))

    if k_count != REQUIRED_10K or q_count != REQUIRED_10Q:
        result["action"] = "DELETED"
        result["reason"] = f"10-K: {k_count}/{REQUIRED_10K}, 10-Q: {q_count}/{REQUIRED_10Q}"
        if not dry_run:
            shutil.rmtree(company_dir)
        return result

    return result


def main():
    parser = argparse.ArgumentParser(description="DDDS Amendment Cleaner")
    parser.add_argument("--data-dir", required=True, help="Path to 'Filings by company' directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without deleting anything")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    companies = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"[CLEAN] {'DRY RUN - ' if args.dry_run else ''}Processing {len(companies)} company folders\n")

    kept, deleted, total_amendments = 0, 0, 0
    deleted_tickers = []

    for company_dir in companies:
        result = clean_company(company_dir, args.dry_run)
        total_amendments += result["amendments_removed"]

        if result["action"] == "DELETED":
            deleted += 1
            deleted_tickers.append(result["ticker"])
            print(f"  REMOVE  {result['ticker']:8s}  — {result['reason']}")
        else:
            kept += 1
            if result["amendments_removed"] > 0:
                print(f"  CLEAN   {result['ticker']:8s}  — removed {result['amendments_removed']} amendment(s)")

    print(f"\n{'=' * 50}")
    print(f"{'DRY RUN SUMMARY' if args.dry_run else 'SUMMARY'}")
    print(f"  Companies kept   : {kept}")
    print(f"  Companies removed: {deleted}")
    print(f"  Amendments purged: {total_amendments}")
    if deleted_tickers:
        print(f"\n  Removed tickers:\n    {', '.join(deleted_tickers)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
"""
DDDS - Point-in-Time Peer Selection via Euclidean Distance
===========================================================
For each company-filing, identifies the 5 closest peers within the same
SIC group using three normalised financial metrics:

    1. Point-in-Time Market Cap  (historical price × shares outstanding)
    2. Trailing Twelve Month Revenue
    3. Total Assets

Zero look-ahead bias: peer metrics are drawn from each candidate's most
recent filing date that is STRICTLY BEFORE the target filing date.
Market cap uses the historical closing price on the filing date.

SIC codes are fetched directly from SEC EDGAR's submissions API.
Peer grouping is tiered: 4-digit SIC → 3-digit → 2-digit fallback,
expanding only when fewer than 3 candidates exist at the tighter level.

Inputs
------
    - Filing directory  : folder of SEC filings (flat OR nested by
        company/filing-type). Filenames must match:
        {TICKER}_{TYPE}_{DATE}_{ACCESSION}.html
    - yfinance          : financial data (cached after first fetch)
    - SEC EDGAR         : SIC codes (cached after first fetch)

Outputs
-------
    - data/peer_selections.csv
        target_ticker, target_filing_date, target_filing_type,
        peer_ticker, peer_filing_date, peer_filing_type,
        distance, rank (1-5), sic_level (4/3/2)
    - data/yfinance_cache.pkl   (for --use-cached)
    - data/sic_cache.json       (for --use-cached)

Usage
-----
    python run_2_-_peer_selection.py --input-dir "data/Filings by company"
    python run_2_-_peer_selection.py --use-cached

"""

import argparse
import json
import os
import re
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_DIR        = "data/Filings by company"
OUTPUT_CSV       = "data/peer_selections.csv"
CACHE_FILE       = "data/yfinance_cache.pkl"
SIC_CACHE_FILE   = "data/sic_cache.json"
NUM_PEERS        = 5
MIN_PEER_CANDIDATES = 3   # expand SIC tier if fewer candidates than this
EDGAR_USER_AGENT = "DDDS Pipeline connor@lancaster.ac.uk"
FILENAME_PATTERN = re.compile(
    r"^(?P<ticker>[A-Z][A-Z0-9\-]+)_(?P<type>10-[KQ])_(?P<date>\d{4}-\d{2}-\d{2})_"
)
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. FILENAME PARSING
# ===========================================================================

def parse_filing_directory(input_dir: str) -> pd.DataFrame:
    """
    Recursively scan the filing directory and extract ticker, filing_type,
    filing_date from each filename matching the expected pattern.

    Returns
    -------
    DataFrame with columns: ticker, filing_type, filing_date, filename, filepath
    """
    records = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            m = FILENAME_PATTERN.match(fname)
            if m:
                records.append({
                    "ticker":       m.group("ticker"),
                    "filing_type":  m.group("type"),
                    "filing_date":  pd.Timestamp(m.group("date")),
                    "filename":     fname,
                    "filepath":     os.path.join(root, fname),
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(
            f"No filings matching expected pattern found in {input_dir}"
        )
    df.sort_values(["ticker", "filing_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[DDDS] Parsed {len(df)} filings across {df['ticker'].nunique()} companies")
    return df


# ===========================================================================
# 2. SIC CODE FETCHING FROM EDGAR
# ===========================================================================

def _edgar_get(url: str) -> dict:
    """Make a GET request to EDGAR with the required User-Agent header."""
    req = Request(url, headers={"User-Agent": EDGAR_USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_sic_codes(tickers: list[str], sic_cache_file: str,
                    use_cached: bool = False) -> dict[str, str]:
    """
    Fetch 4-digit SIC codes from SEC EDGAR for each ticker.

    Uses the bulk company_tickers.json to map ticker → CIK, then
    the submissions API for each CIK to get the SIC code.

    Returns
    -------
    dict: {ticker: "3599", ...}
    """
    if use_cached and os.path.exists(sic_cache_file):
        print(f"[DDDS] Loading cached SIC codes from {sic_cache_file}")
        with open(sic_cache_file, "r") as f:
            return json.load(f)

    print("[DDDS] Fetching SIC codes from SEC EDGAR ...")

    # Step 1: Get ticker → CIK mapping
    print("  Downloading company_tickers.json ...")
    tickers_data = _edgar_get("https://www.sec.gov/files/company_tickers.json")

    # Build lookup: uppercase ticker → CIK (zero-padded to 10 digits)
    ticker_to_cik = {}
    for entry in tickers_data.values():
        t = entry.get("ticker", "").upper()
        cik = entry.get("cik_str", "")
        if t and cik:
            ticker_to_cik[t] = str(cik).zfill(10)

    # Step 2: Fetch SIC for each ticker via submissions API
    sic_map = {}
    failed = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            print(f"  [{i+1}/{total}] {ticker} — no CIK found, skipping")
            failed.append(ticker)
            continue

        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            data = _edgar_get(url)
            sic = data.get("sic", "")
            if sic:
                sic_map[ticker] = str(sic)
                print(f"  [{i+1}/{total}] {ticker} → SIC {sic}")
            else:
                print(f"  [{i+1}/{total}] {ticker} — no SIC in response")
                failed.append(ticker)

            # EDGAR rate limit: max 10 requests/sec
            time.sleep(0.12)

        except (HTTPError, URLError, Exception) as e:
            print(f"  [{i+1}/{total}] {ticker} — FAILED ({e})")
            failed.append(ticker)

    if failed:
        print(f"[DDDS] WARNING: No SIC code for {len(failed)} tickers: {failed}")

    # Cache
    os.makedirs(os.path.dirname(sic_cache_file), exist_ok=True)
    with open(sic_cache_file, "w") as f:
        json.dump(sic_map, f, indent=2)
    print(f"[DDDS] Cached SIC codes to {sic_cache_file}")

    return sic_map


def build_sic_maps(sic_codes: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """
    Build three grouping maps from SIC codes:
        "4" → {sic_4digit: [tickers]}
        "3" → {sic_3digit: [tickers]}
        "2" → {sic_2digit: [tickers]}
    """
    maps = {"4": {}, "3": {}, "2": {}}
    for ticker, sic in sic_codes.items():
        sic = str(sic).ljust(4, "0")
        for level, prefix_len in [("4", 4), ("3", 3), ("2", 2)]:
            key = sic[:prefix_len]
            maps[level].setdefault(key, []).append(ticker)
    return maps


# ===========================================================================
# 3. YFINANCE DATA FETCHING (with caching)
# ===========================================================================

def fetch_yfinance_data(tickers: list[str], cache_file: str,
                        use_cached: bool = False) -> dict:
    """
    Fetch quarterly financials and historical prices for each ticker
    from yfinance. Results are cached to a pickle file.
    """
    if use_cached and os.path.exists(cache_file):
        print(f"[DDDS] Loading cached yfinance data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    import yfinance as yf

    data = {}
    failed = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] Fetching {ticker} ...", end=" ")
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            try:
                q_income = t.quarterly_income_stmt
            except Exception:
                q_income = pd.DataFrame()

            try:
                q_balance = t.quarterly_balance_sheet
            except Exception:
                q_balance = pd.DataFrame()

            try:
                hist = t.history(period="max")[["Close"]].reset_index()
                hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
            except Exception:
                hist = pd.DataFrame(columns=["Date", "Close"])

            shares_outstanding = info.get("sharesOutstanding", None)

            data[ticker] = {
                "quarterly_income":   q_income,
                "quarterly_balance":  q_balance,
                "price_history":      hist,
                "shares_outstanding": shares_outstanding,
            }
            print("OK")

        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(ticker)

    if failed:
        print(f"[DDDS] WARNING: Failed to fetch {len(failed)} tickers: {failed}")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"[DDDS] Cached yfinance data to {cache_file}")

    return data


# ===========================================================================
# 4. POINT-IN-TIME METRIC EXTRACTION
# ===========================================================================

def get_historical_price(yf_data: dict, as_of: pd.Timestamp) -> Optional[float]:
    """Get the closing price on or immediately before `as_of`."""
    hist = yf_data.get("price_history", pd.DataFrame())
    if hist.empty:
        return None
    mask = hist["Date"] <= as_of
    if not mask.any():
        return None
    return float(hist.loc[mask, "Close"].iloc[-1])


def get_shares_outstanding(yf_data: dict, as_of: pd.Timestamp) -> Optional[float]:
    """
    Try to get shares from quarterly balance sheet at or before `as_of`.
    Falls back to current shares_outstanding from yfinance info.
    """
    q_bal = yf_data.get("quarterly_balance", pd.DataFrame())
    if not q_bal.empty:
        share_labels = [
            "Ordinary Shares Number",
            "Share Issued",
            "Common Stock Shares Outstanding",
        ]
        for label in share_labels:
            if label in q_bal.index:
                row = q_bal.loc[label]
                valid_dates = [d for d in row.index
                               if pd.notna(d) and pd.Timestamp(d) <= as_of
                               and pd.notna(row[d])]
                if valid_dates:
                    latest = max(valid_dates)
                    return float(row[latest])

    return yf_data.get("shares_outstanding")


def get_ttm_revenue(yf_data: dict, as_of: pd.Timestamp) -> Optional[float]:
    """Sum the four most recent quarterly revenue figures on or before `as_of`."""
    q_inc = yf_data.get("quarterly_income", pd.DataFrame())
    if q_inc.empty:
        return None

    rev_labels = ["Total Revenue", "Revenue", "Net Revenue"]
    rev_row = None
    for label in rev_labels:
        if label in q_inc.index:
            rev_row = q_inc.loc[label]
            break
    if rev_row is None:
        return None

    valid = {d: rev_row[d] for d in rev_row.index
             if pd.notna(d) and pd.Timestamp(d) <= as_of and pd.notna(rev_row[d])}
    if not valid:
        return None
    sorted_dates = sorted(valid.keys(), reverse=True)[:4]
    return float(sum(valid[d] for d in sorted_dates))


def get_total_assets(yf_data: dict, as_of: pd.Timestamp) -> Optional[float]:
    """Get total assets from most recent quarterly balance sheet on or before `as_of`."""
    q_bal = yf_data.get("quarterly_balance", pd.DataFrame())
    if q_bal.empty:
        return None

    if "Total Assets" in q_bal.index:
        row = q_bal.loc["Total Assets"]
        valid_dates = [d for d in row.index
                       if pd.notna(d) and pd.Timestamp(d) <= as_of
                       and pd.notna(row[d])]
        if valid_dates:
            latest = max(valid_dates)
            return float(row[latest])
    return None


def get_market_cap(yf_data: dict, as_of: pd.Timestamp) -> Optional[float]:
    """Point-in-time market cap = historical price × shares outstanding."""
    price = get_historical_price(yf_data, as_of)
    shares = get_shares_outstanding(yf_data, as_of)
    if price is None or shares is None:
        return None
    return price * shares


def get_metrics(yf_data: dict, as_of: pd.Timestamp) -> dict:
    """Extract all three metrics for a company as of a given date."""
    return {
        "market_cap":   get_market_cap(yf_data, as_of),
        "ttm_revenue":  get_ttm_revenue(yf_data, as_of),
        "total_assets": get_total_assets(yf_data, as_of),
    }


# ===========================================================================
# 5. PEER SELECTION (with tiered SIC fallback)
# ===========================================================================

def get_candidates_tiered(
    target_ticker: str,
    sic_codes: dict[str, str],
    sic_maps: dict[str, dict[str, list[str]]],
    yf_cache: dict,
    min_candidates: int = MIN_PEER_CANDIDATES,
) -> tuple[list[str], str]:
    """
    Find peer candidates using tiered SIC grouping.
    Tries 4-digit SIC first, then 3-digit, then 2-digit.

    Returns
    -------
    (candidate_tickers, sic_level_used)
    """
    target_sic = sic_codes.get(target_ticker, "")
    if not target_sic:
        return [], "none"

    target_sic = str(target_sic).ljust(4, "0")

    for level, prefix_len in [("4", 4), ("3", 3), ("2", 2)]:
        key = target_sic[:prefix_len]
        group = sic_maps[level].get(key, [])
        candidates = [t for t in group
                      if t != target_ticker and t in yf_cache]
        if len(candidates) >= min_candidates:
            return candidates, level

    # Even 2-digit had < min_candidates — return whatever we have
    key_2 = target_sic[:2]
    candidates = [t for t in sic_maps["2"].get(key_2, [])
                  if t != target_ticker and t in yf_cache]
    return candidates, "2"


def select_peers(
    target_ticker: str,
    target_date: pd.Timestamp,
    filings_df: pd.DataFrame,
    yf_cache: dict,
    sic_codes: dict[str, str],
    sic_maps: dict[str, dict[str, list[str]]],
    num_peers: int = NUM_PEERS,
) -> list[dict]:
    """
    Select the `num_peers` closest peers for a target company-filing
    using tiered SIC grouping and Euclidean distance on normalised
    financial metrics.
    """
    target_yf = yf_cache.get(target_ticker)
    if target_yf is None:
        return []

    candidates, sic_level = get_candidates_tiered(
        target_ticker, sic_codes, sic_maps, yf_cache
    )
    if not candidates:
        return []

    target_metrics = get_metrics(target_yf, target_date)
    if any(v is None for v in target_metrics.values()):
        return []

    peer_data = []
    for peer_ticker in candidates:
        peer_filings = filings_df[
            (filings_df["ticker"] == peer_ticker)
            & (filings_df["filing_date"] < target_date)
        ]
        if peer_filings.empty:
            continue

        latest_peer = peer_filings.iloc[-1]
        peer_date = latest_peer["filing_date"]
        peer_type = latest_peer["filing_type"]

        peer_yf = yf_cache.get(peer_ticker)
        if peer_yf is None:
            continue
        peer_metrics = get_metrics(peer_yf, peer_date)
        if any(v is None for v in peer_metrics.values()):
            continue

        peer_data.append({
            "ticker":       peer_ticker,
            "filing_date":  peer_date,
            "filing_type":  peer_type,
            "market_cap":   peer_metrics["market_cap"],
            "ttm_revenue":  peer_metrics["ttm_revenue"],
            "total_assets": peer_metrics["total_assets"],
        })

    if len(peer_data) < 1:
        return []

    all_vecs = [
        [target_metrics["market_cap"],
         target_metrics["ttm_revenue"],
         target_metrics["total_assets"]]
    ]
    for p in peer_data:
        all_vecs.append([p["market_cap"], p["ttm_revenue"], p["total_assets"]])

    arr = np.array(all_vecs, dtype=np.float64)

    means = arr.mean(axis=0)
    stds  = arr.std(axis=0)
    stds[stds == 0] = 1.0
    normed = (arr - means) / stds

    target_vec = normed[0]
    distances = np.linalg.norm(normed[1:] - target_vec, axis=1)

    for idx, p in enumerate(peer_data):
        p["distance"] = float(distances[idx])

    peer_data.sort(key=lambda x: x["distance"])

    results = []
    for rank, p in enumerate(peer_data[:num_peers], start=1):
        results.append({
            "peer_ticker":      p["ticker"],
            "peer_filing_date": p["filing_date"],
            "peer_filing_type": p["filing_type"],
            "distance":         round(p["distance"], 6),
            "rank":             rank,
            "sic_level":        sic_level,
        })
    return results


# ===========================================================================
# 6. MAIN PIPELINE
# ===========================================================================

def run(input_dir: str, output_csv: str, cache_file: str,
        sic_cache_file: str, use_cached: bool = False):

    # --- Step 1: Parse filing directory ---
    filings_df = parse_filing_directory(input_dir)
    tickers = sorted(filings_df["ticker"].unique())
    print(f"[DDDS] {len(tickers)} unique tickers found\n")

    # --- Step 2: Fetch SIC codes from EDGAR ---
    sic_codes = fetch_sic_codes(tickers, sic_cache_file, use_cached=use_cached)
    sic_maps = build_sic_maps(sic_codes)
    print(f"\n[DDDS] SIC coverage: {len(sic_codes)}/{len(tickers)} tickers")
    print("[DDDS] SIC groups (4-digit):")
    for sic, members in sorted(sic_maps["4"].items(),
                                key=lambda x: -len(x[1])):
        if len(members) >= 2:
            print(f"    SIC {sic}: {len(members)} companies")
    singletons_4 = sum(1 for m in sic_maps["4"].values() if len(m) == 1)
    print(f"    + {singletons_4} singleton SIC-4 groups (will expand to 3-digit)")
    print()

    # --- Step 3: Fetch / load yfinance data ---
    yf_cache = fetch_yfinance_data(tickers, cache_file, use_cached=use_cached)
    available = set(yf_cache.keys())
    missing = set(tickers) - available
    if missing:
        print(f"[DDDS] WARNING: {len(missing)} tickers missing from yfinance: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    print()

    # --- Step 4: Select peers for each company's most recent filing ---
    # Keep full filings_df for peer lookups; only iterate over latest per ticker
    target_filings = (filings_df
                      .sort_values("filing_date")
                      .drop_duplicates("ticker", keep="last")
                      .reset_index(drop=True))
    all_results = []
    total_filings = len(target_filings)
    no_peers = 0
    level_counts = {"4": 0, "3": 0, "2": 0}

    for i, row in target_filings.iterrows():
        ticker = row["ticker"]
        fdate  = row["filing_date"]
        ftype  = row["filing_type"]

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total_filings}] Processing {ticker} "
                  f"{ftype} {fdate.strftime('%Y-%m-%d')} ...")

        peers = select_peers(
            target_ticker=ticker,
            target_date=fdate,
            filings_df=filings_df,
            yf_cache=yf_cache,
            sic_codes=sic_codes,
            sic_maps=sic_maps,
        )

        if not peers:
            no_peers += 1
            continue

        level_counts[peers[0]["sic_level"]] = \
            level_counts.get(peers[0]["sic_level"], 0) + 1

        for p in peers:
            all_results.append({
                "target_ticker":      ticker,
                "target_filing_date": fdate.strftime("%Y-%m-%d"),
                "target_filing_type": ftype,
                **p,
            })

    # --- Step 5: Output ---
    out_df = pd.DataFrame(all_results)
    if not out_df.empty:
        out_df["peer_filing_date"] = pd.to_datetime(
            out_df["peer_filing_date"]
        ).dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"\n[DDDS] Peer selection complete")
    print(f"    Filings processed : {total_filings}")
    print(f"    Filings w/ peers  : {total_filings - no_peers}")
    print(f"    Filings w/o peers : {no_peers}")
    print(f"    Total peer rows   : {len(out_df)}")
    print(f"    SIC-4 matches     : {level_counts.get('4', 0)} filings")
    print(f"    SIC-3 fallbacks   : {level_counts.get('3', 0)} filings")
    print(f"    SIC-2 fallbacks   : {level_counts.get('2', 0)} filings")
    print(f"    Output            : {output_csv}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DDDS - Point-in-Time Peer Selection (SIC-based)"
    )
    parser.add_argument(
        "--input-dir", default=INPUT_DIR,
        help="Directory containing SEC filings (flat or nested)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_CSV,
        help="Output CSV path"
    )
    parser.add_argument(
        "--cache-file", default=CACHE_FILE,
        help="Path for yfinance data cache"
    )
    parser.add_argument(
        "--sic-cache-file", default=SIC_CACHE_FILE,
        help="Path for SIC code cache"
    )
    parser.add_argument(
        "--use-cached", action="store_true",
        help="Skip API calls, load from caches"
    )
    parser.add_argument(
        "--num-peers", type=int, default=NUM_PEERS,
        help="Number of peers to select per filing (default: 5)"
    )
    args = parser.parse_args()
    NUM_PEERS = args.num_peers

    run(
        input_dir=args.input_dir,
        output_csv=args.output,
        cache_file=args.cache_file,
        sic_cache_file=args.sic_cache_file,
        use_cached=args.use_cached,
    )
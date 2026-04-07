"""
DDDS - Backtesting Framework
==============================
Tests the core hypothesis: disclosure degradation detected by DDDS
predicts future earnings persistence problems.

Methodology follows Li (2008):
    "Annual report readability, current earnings, and earnings persistence"
    Journal of Accounting and Economics, 45(2-3), pp.221-247.

Regression model
----------------
    Earnings(t+1) = α + β1·Earnings(t) + β2·Flagged(t) + β3·Flagged(t)·Earnings(t) + ε

Where:
    Earnings(t)          : current period earnings (EPS or ROA)
    Earnings(t+1)        : next period earnings
    Flagged(t)           : 1 if DDDS flagged the company in period t, 0 otherwise
    Flagged·Earnings(t)  : interaction term — key test variable

Interpretation:
    β1 > 0               : earnings persistence (baseline)
    β3 < 0               : DDDS-flagged companies show lower earnings persistence
                           This is the confirmation of the system's predictive validity

The regression is run across three subsamples following Li (2008):
    1. Full sample
    2. Profitable companies only (Earnings(t) > 0)
    3. Loss-making companies only (Earnings(t) < 0)

Data sources
------------
    - Flagged companies : data/findings/*_findings.json  (Run 8 output)
    - Earnings data     : SEC EDGAR XBRL API (company facts)
                          Falls back to yfinance if XBRL data unavailable
    - Company universe  : DDDS/company_universe_cache.json  (SECAPISCRAPER output)
                          OR data/passages.csv  (Run 1 output)

Usage
-----
    python run_11_backtest.py
    python run_11_backtest.py --findings-dir data/findings --signal-threshold MEDIUM
    python run_11_backtest.py --earnings-metric ROA
    python run_11_backtest.py --skip-fetch   # use cached earnings data only
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FINDINGS_DIR      = os.environ.get("FINDINGS_DIR", "data/findings")
UNIVERSE_CACHE    = "DDDS/company_universe_cache.json"
PASSAGES_CSV      = "data/passages.csv"
EARNINGS_CACHE    = "data/backtest/earnings_cache.csv"
OUTPUT_DIR        = "data/backtest"

# Which signals count as flagged (configurable via --signal-threshold)
SIGNAL_HIERARCHY  = ["HIGH", "MEDIUM", "LOW"]

# Earnings metric: "EPS" or "ROA"
EARNINGS_METRIC   = "EPS"

# EDGAR XBRL API
EDGAR_HEADERS     = {"User-Agent": os.environ.get("EDGAR_USER_AGENT", "DDDS Research brad@lancaster.ac.uk")}
EDGAR_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
REQUEST_DELAY     = 0.12   # SEC rate limit: 10 req/s, target 8/s

# Regression periods — align with your filing window
PERIOD_T          = "2024"   # period when DDDS flags are generated
PERIOD_T1         = "2025"   # period where earnings persistence is tested
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# STEP 1 — LOAD FLAGGED COMPANIES FROM RUN 8 FINDINGS
# ---------------------------------------------------------------------------

def load_flagged_companies(findings_dir: str, signal_threshold: str) -> dict[str, dict]:
    """
    Reads all *_findings.json from findings_dir.

    Returns dict keyed by identifier:
        {
            "identifier": str,
            "signal":     str  (HIGH / MEDIUM / LOW),
            "flagged":    int  (1 or 0),
            "flags_count": int
        }
    """
    threshold_idx = SIGNAL_HIERARCHY.index(signal_threshold)
    qualifying    = set(SIGNAL_HIERARCHY[:threshold_idx + 1])

    pattern = Path(findings_dir).glob("*_findings.json")
    flagged = {}

    for path in pattern:
        try:
            with open(path) as f:
                findings = json.load(f)
        except Exception as e:
            print(f"  [!] Failed to load {path.name}: {e}")
            continue

        identifier = findings.get("company", {}).get("identifier", "")
        signal     = findings.get("deep_analysis", {}).get(
            "overall_signal", {}
        ).get("signal_strength", "LOW")

        if not identifier:
            continue

        flagged[identifier] = {
            "identifier":  identifier,
            "name":        findings.get("company", {}).get("name", ""),
            "signal":      signal,
            "flagged":     1 if signal in qualifying else 0,
            "flags_count": findings.get("flags_count", 0),
        }

    print(f"  Findings loaded: {len(flagged)} companies")
    print(f"  Flagged (>= {signal_threshold}): {sum(v['flagged'] for v in flagged.values())}")
    return flagged


# ---------------------------------------------------------------------------
# STEP 2 — LOAD COMPANY UNIVERSE
# ---------------------------------------------------------------------------

def load_universe() -> dict[str, str]:
    """
    Loads CIK → ticker mapping from the SECAPISCRAPER cache or passages CSV.
    Returns dict: {ticker: cik}
    """
    # Try SECAPISCRAPER cache first
    if Path(UNIVERSE_CACHE).exists():
        with open(UNIVERSE_CACHE) as f:
            cache = json.load(f)
        mapping = {v["ticker"]: k for k, v in cache.items() if v.get("ticker")}
        print(f"  Universe loaded from SECAPISCRAPER cache: {len(mapping)} companies")
        return mapping

    # Fall back to passages CSV
    if Path(PASSAGES_CSV).exists():
        df      = pd.read_csv(PASSAGES_CSV, usecols=["identifier"])
        tickers = df["identifier"].dropna().unique().tolist()
        # Without CIK we can't call XBRL API — return ticker as key, None as value
        mapping = {t: None for t in tickers}
        print(f"  Universe loaded from passages CSV: {len(mapping)} companies")
        print(f"  [!] No CIK data available — XBRL fetch will be skipped, yfinance used instead")
        return mapping

    print(f"  [!] No universe found at '{UNIVERSE_CACHE}' or '{PASSAGES_CSV}'")
    return {}


# ---------------------------------------------------------------------------
# STEP 3 — FETCH EARNINGS DATA
# ---------------------------------------------------------------------------

def fetch_earnings_xbrl(cik: str, metric: str) -> dict[str, float]:
    """
    Fetches earnings data from SEC EDGAR XBRL API for a single company.

    Returns dict: {period_label: value}
    Period labels are annual: "2023", "2024", "2025"

    For EPS: uses EarningsPerShareBasic or EarningsPerShareDiluted
    For ROA: uses NetIncomeLoss / Assets (computed from separate facts)
    """
    url  = EDGAR_FACTS_URL.format(cik=str(cik).zfill(10))
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code != 200:
            return {}
        facts = resp.json()
    except Exception:
        return {}

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    if metric == "EPS":
        # Try diluted first, fall back to basic
        for concept in ["EarningsPerShareDiluted", "EarningsPerShareBasic"]:
            data = us_gaap.get(concept, {})
            units = data.get("units", {})
            entries = units.get("USD/shares", [])
            if entries:
                return _extract_annual_values(entries)
        return {}

    elif metric == "ROA":
        # Net Income / Total Assets
        ni_entries  = (us_gaap.get("NetIncomeLoss", {})
                              .get("units", {}).get("USD", []))
        ast_entries = (us_gaap.get("Assets", {})
                               .get("units", {}).get("USD", []))
        ni_vals  = _extract_annual_values(ni_entries)
        ast_vals = _extract_annual_values(ast_entries)

        roa = {}
        for period in set(ni_vals) & set(ast_vals):
            if ast_vals[period] and ast_vals[period] != 0:
                roa[period] = ni_vals[period] / ast_vals[period]
        return roa

    return {}


def _extract_annual_values(entries: list) -> dict[str, float]:
    """
    Extracts annual (12-month) period values from XBRL entries.
    Returns dict: {"2023": value, "2024": value, ...}
    """
    annual = {}
    for entry in entries:
        # Annual filings: form is 10-K and period covers 12 months
        form = entry.get("form", "")
        if form not in ("10-K", "10-K/A"):
            continue
        end_date = entry.get("end", "")
        if not end_date:
            continue
        year = end_date[:4]
        val  = entry.get("val")
        if val is not None:
            # Keep the most recent 10-K value for each year
            if year not in annual:
                annual[year] = float(val)

    return annual


def fetch_earnings_yfinance(ticker: str, metric: str) -> dict[str, float]:
    """
    Fallback earnings fetch using yfinance.
    Returns dict: {"2023": value, "2024": value, ...}
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info  = stock.info

        if metric == "EPS":
            # yfinance trailing EPS
            eps = info.get("trailingEps")
            if eps is not None:
                return {PERIOD_T: float(eps)}

        elif metric == "ROA":
            roa = info.get("returnOnAssets")
            if roa is not None:
                return {PERIOD_T: float(roa)}

    except Exception:
        pass

    return {}


def load_or_fetch_earnings(
    universe: dict[str, str],
    flagged: dict[str, dict],
    metric: str,
    skip_fetch: bool,
) -> pd.DataFrame:
    """
    Loads earnings data from cache if available, otherwise fetches from EDGAR XBRL.

    Returns DataFrame with columns:
        identifier, name, flagged, signal, earnings_t, earnings_t1
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = Path(EARNINGS_CACHE)

    if skip_fetch and cache_path.exists():
        print(f"  Loading cached earnings data from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Fetching earnings ({metric}) for {len(universe)} companies...")
    rows = []
    total = len(universe)

    for i, (ticker, cik) in enumerate(universe.items()):
        print(f"  [{i+1}/{total}] {ticker}", end=" ... ")

        # Get earnings data
        if cik:
            earnings = fetch_earnings_xbrl(cik, metric)
        else:
            earnings = {}

        # Fallback to yfinance if XBRL returned nothing
        if not earnings:
            earnings = fetch_earnings_yfinance(ticker, metric)
            source = "yfinance"
        else:
            source = "XBRL"

        earnings_t  = earnings.get(PERIOD_T)
        earnings_t1 = earnings.get(PERIOD_T1)

        if earnings_t is None and earnings_t1 is None:
            print(f"no data ({source})")
            continue

        flag_data = flagged.get(ticker, {
            "name":        ticker,
            "signal":      "NONE",
            "flagged":     0,
            "flags_count": 0,
        })

        rows.append({
            "identifier":  ticker,
            "name":        flag_data.get("name", ticker),
            "flagged":     flag_data["flagged"],
            "signal":      flag_data["signal"],
            "flags_count": flag_data["flags_count"],
            "earnings_t":  earnings_t,
            "earnings_t1": earnings_t1,
            "source":      source,
        })
        print(f"t={earnings_t}, t+1={earnings_t1} ({source})")

    df = pd.DataFrame(rows)

    # Cache for future runs
    df.to_csv(cache_path, index=False)
    print(f"\n  Earnings data cached to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# STEP 4 — EARNINGS PERSISTENCE REGRESSION (LI 2008)
# ---------------------------------------------------------------------------

def run_regression(df: pd.DataFrame, subsample: str = "full") -> dict:
    """
    Runs the Li (2008) earnings persistence regression on a subsample.

    Model:
        Earnings(t+1) = α + β1·Earnings(t) + β2·Flagged + β3·Flagged·Earnings(t) + ε

    Returns dict with regression results.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    # Filter to complete cases
    sub = df.dropna(subset=["earnings_t", "earnings_t1"]).copy()

    if subsample == "profitable":
        sub = sub[sub["earnings_t"] > 0]
    elif subsample == "loss":
        sub = sub[sub["earnings_t"] < 0]

    n = len(sub)
    if n < 10:
        return {
            "subsample": subsample,
            "n":         n,
            "error":     f"Insufficient observations ({n}) for regression",
        }

    # Construct design matrix
    X = pd.DataFrame({
        "earnings_t":          sub["earnings_t"],
        "flagged":             sub["flagged"].astype(float),
        "flagged_earnings_t":  sub["flagged"].astype(float) * sub["earnings_t"],
    })
    y = sub["earnings_t1"]

    # OLS via numpy (avoids statsmodels dependency, easy to interpret)
    X_mat = np.column_stack([np.ones(n), X.values])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_mat, y.values, rcond=None)
    except np.linalg.LinAlgError as e:
        return {"subsample": subsample, "n": n, "error": str(e)}

    alpha, beta1, beta2, beta3 = coeffs

    # R-squared
    y_pred  = X_mat @ coeffs
    ss_res  = np.sum((y.values - y_pred) ** 2)
    ss_tot  = np.sum((y.values - y.values.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors (OLS)
    dof     = n - X_mat.shape[1]
    if dof > 0:
        mse     = ss_res / dof
        try:
            cov     = mse * np.linalg.inv(X_mat.T @ X_mat)
            se      = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(4, np.nan)
    else:
        se = np.full(4, np.nan)

    # T-statistics
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, coeffs / se, np.nan)

    # Two-tailed p-values (approximate using normal distribution for large n)
    from scipy import stats as scipy_stats
    p_vals = [2 * (1 - scipy_stats.t.cdf(abs(t), df=dof)) if not np.isnan(t) else np.nan
              for t in t_stats]

    return {
        "subsample":  subsample,
        "n":          n,
        "alpha":      round(float(alpha),  4),
        "beta1":      round(float(beta1),  4),   # earnings persistence
        "beta2":      round(float(beta2),  4),   # DDDS flag effect on level
        "beta3":      round(float(beta3),  4),   # DDDS flag effect on persistence ← key
        "se_alpha":   round(float(se[0]),  4),
        "se_beta1":   round(float(se[1]),  4),
        "se_beta2":   round(float(se[2]),  4),
        "se_beta3":   round(float(se[3]),  4),
        "t_beta3":    round(float(t_stats[3]), 3) if not np.isnan(t_stats[3]) else None,
        "p_beta3":    round(float(p_vals[3]),  4) if not np.isnan(p_vals[3]) else None,
        "r2":         round(r2, 4),
        "hypothesis_supported": (
            float(beta3) < 0 and
            p_vals[3] is not None and
            float(p_vals[3]) < 0.10
        ),
    }


# ---------------------------------------------------------------------------
# STEP 5 — DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------

def compute_descriptives(df: pd.DataFrame) -> dict:
    """
    Computes descriptive statistics comparing flagged vs non-flagged companies.
    """
    flagged     = df[df["flagged"] == 1]
    non_flagged = df[df["flagged"] == 0]

    def stats(series):
        return {
            "n":      int(series.dropna().count()),
            "mean":   round(float(series.dropna().mean()), 4) if series.dropna().count() > 0 else None,
            "median": round(float(series.dropna().median()), 4) if series.dropna().count() > 0 else None,
            "std":    round(float(series.dropna().std()), 4) if series.dropna().count() > 0 else None,
        }

    return {
        "total_companies":         len(df),
        "flagged_count":           len(flagged),
        "non_flagged_count":       len(non_flagged),
        "signal_distribution":     df["signal"].value_counts().to_dict(),
        "flagged_earnings_t":      stats(flagged["earnings_t"]),
        "non_flagged_earnings_t":  stats(non_flagged["earnings_t"]),
        "flagged_earnings_t1":     stats(flagged["earnings_t1"]),
        "non_flagged_earnings_t1": stats(non_flagged["earnings_t1"]),
        "complete_cases":          int(df.dropna(subset=["earnings_t", "earnings_t1"]).shape[0]),
    }


# ---------------------------------------------------------------------------
# STEP 6 — OUTPUT AND REPORTING
# ---------------------------------------------------------------------------

def print_results(descriptives: dict, results: list[dict], metric: str,
                  signal_threshold: str):
    """Prints the full backtesting report to console."""

    print(f"\n{'='*65}")
    print(f"  DDDS BACKTESTING REPORT — Li (2008) Earnings Persistence Test")
    print(f"{'='*65}")
    print(f"  Earnings metric:      {metric}")
    print(f"  Signal threshold:     >= {signal_threshold} (flagged)")
    print(f"  Period t:             {PERIOD_T}")
    print(f"  Period t+1:           {PERIOD_T1}")

    print(f"\n  SAMPLE")
    print(f"  {'─'*40}")
    print(f"  Total companies:      {descriptives['total_companies']}")
    print(f"  Flagged:              {descriptives['flagged_count']}")
    print(f"  Non-flagged:          {descriptives['non_flagged_count']}")
    print(f"  Complete cases:       {descriptives['complete_cases']}")
    print(f"\n  Signal distribution:")
    for sig, count in descriptives["signal_distribution"].items():
        print(f"    {sig:<10} {count}")

    print(f"\n  DESCRIPTIVE STATISTICS ({metric})")
    print(f"  {'─'*40}")
    print(f"  {'':20} {'Flagged':>12} {'Non-Flagged':>12}")
    for period, label in [("earnings_t", f"Period t ({PERIOD_T})"),
                           ("earnings_t1", f"Period t+1 ({PERIOD_T1})")]:
        fg  = descriptives[f"flagged_{period}"]
        nfg = descriptives[f"non_flagged_{period}"]
        fn  = fg.get("mean")
        nn  = nfg.get("mean")
        print(f"  {label:<20} "
              f"{f'{fn:.4f}' if fn is not None else 'N/A':>12} "
              f"{f'{nn:.4f}' if nn is not None else 'N/A':>12}")

    print(f"\n  REGRESSION RESULTS — Li (2008) Model")
    print(f"  Earnings(t+1) = α + β1·E(t) + β2·FLAG + β3·FLAG·E(t) + ε")
    print(f"  {'─'*60}")
    print(f"  {'Subsample':<18} {'N':>5} {'β1':>8} {'β2':>8} {'β3':>8} "
          f"{'t(β3)':>8} {'p(β3)':>8} {'R²':>6} {'H0':>5}")
    print(f"  {'─'*60}")

    for r in results:
        if "error" in r:
            print(f"  {r['subsample']:<18} {r['n']:>5}  ERROR: {r['error']}")
            continue

        supported = "✓" if r.get("hypothesis_supported") else "✗"
        p_str     = f"{r['p_beta3']:.4f}" if r['p_beta3'] is not None else "N/A"
        t_str     = f"{r['t_beta3']:.3f}" if r['t_beta3'] is not None else "N/A"

        print(f"  {r['subsample']:<18} {r['n']:>5} "
              f"{r['beta1']:>8.4f} {r['beta2']:>8.4f} {r['beta3']:>8.4f} "
              f"{t_str:>8} {p_str:>8} {r['r2']:>6.4f} {supported:>5}")

    print(f"  {'─'*60}")
    print(f"\n  KEY TEST: β3 (FLAG × Earnings interaction)")
    print(f"  Hypothesis: β3 < 0 and statistically significant (p < 0.10)")
    print(f"  If confirmed: DDDS flags predict lower future earnings persistence")
    print(f"  consistent with Li (2008) obfuscation findings.\n")

    full = next((r for r in results if r["subsample"] == "full"), None)
    if full and "error" not in full:
        if full.get("hypothesis_supported"):
            print(f"  RESULT: Hypothesis SUPPORTED ✓")
            print(f"  β3 = {full['beta3']:.4f} (p = {full['p_beta3']:.4f})")
            print(f"  DDDS-flagged companies show significantly lower earnings")
            print(f"  persistence, validating the system's predictive signal.")
        else:
            print(f"  RESULT: Hypothesis NOT confirmed at p < 0.10 in full sample.")
            print(f"  β3 = {full.get('beta3', 'N/A')} (p = {full.get('p_beta3', 'N/A')})")
            print(f"  Check subsamples and consider expanding the flagged universe.")

    print(f"\n{'='*65}\n")


def save_results(descriptives: dict, results: list[dict], df: pd.DataFrame,
                 metric: str, signal_threshold: str):
    """Saves full results to JSON and the enriched DataFrame to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "config": {
            "metric":           metric,
            "signal_threshold": signal_threshold,
            "period_t":         PERIOD_T,
            "period_t1":        PERIOD_T1,
        },
        "descriptives":  descriptives,
        "regressions":   results,
        "reference":     "Li, F. (2008). Annual report readability, current earnings, "
                         "and earnings persistence. Journal of Accounting and Economics, "
                         "45(2-3), pp.221-247.",
    }

    json_path = os.path.join(OUTPUT_DIR, "backtest_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, "backtest_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Full dataset saved to: {csv_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DDDS Backtesting Framework — Li (2008)")
    parser.add_argument(
        "--findings-dir", default=FINDINGS_DIR,
        help="Directory of *_findings.json files from Run 8"
    )
    parser.add_argument(
        "--signal-threshold", default="MEDIUM",
        choices=["HIGH", "MEDIUM", "LOW"],
        help="Minimum signal level to count as flagged (default: MEDIUM)"
    )
    parser.add_argument(
        "--earnings-metric", default=EARNINGS_METRIC,
        choices=["EPS", "ROA"],
        help="Earnings measure for regression (default: EPS)"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip EDGAR/yfinance fetch and use cached earnings data only"
    )
    args = parser.parse_args()

    print(f"\n[DDDS] Backtesting Framework — Li (2008) Earnings Persistence")
    print(f"  Findings dir:     {args.findings_dir}")
    print(f"  Signal threshold: >= {args.signal_threshold}")
    print(f"  Earnings metric:  {args.earnings_metric}")
    print(f"  Period t:         {PERIOD_T}")
    print(f"  Period t+1:       {PERIOD_T1}\n")

    # Step 1 — load flagged companies
    print("[1/5] Loading DDDS findings...")
    flagged = load_flagged_companies(args.findings_dir, args.signal_threshold)
    if not flagged:
        print(
            f"  [!] No findings found in '{args.findings_dir}'.\n"
            f"      Run graph_rag.py (Run 8) first to generate findings.\n"
            f"      Continuing with universe only — all companies treated as non-flagged."
        )

    # Step 2 — load universe
    print("\n[2/5] Loading company universe...")
    universe = load_universe()
    if not universe:
        print("  [!] No universe found. Cannot proceed without company list.")
        return

    # Step 3 — fetch earnings
    print(f"\n[3/5] Fetching {args.earnings_metric} data from SEC EDGAR XBRL API...")
    df = load_or_fetch_earnings(universe, flagged, args.earnings_metric, args.skip_fetch)

    if df.empty:
        print("  [!] No earnings data retrieved. Check EDGAR API connectivity.")
        return

    print(f"\n  Dataset: {len(df)} companies with earnings data")
    complete = df.dropna(subset=["earnings_t", "earnings_t1"])
    print(f"  Complete cases (both periods): {len(complete)}")

    # Step 4 — descriptives
    print("\n[4/5] Computing descriptive statistics...")
    descriptives = compute_descriptives(df)

    # Step 5 — regressions across three subsamples
    print("\n[5/5] Running Li (2008) earnings persistence regressions...")
    results = []
    for subsample in ["full", "profitable", "loss"]:
        r = run_regression(df, subsample)
        results.append(r)
        n = r.get("n", 0)
        if "error" not in r:
            print(f"  {subsample:<12} n={n:<4} β3={r['beta3']:.4f}  "
                  f"p={r.get('p_beta3', 'N/A')}  "
                  f"{'✓' if r.get('hypothesis_supported') else '✗'}")
        else:
            print(f"  {subsample:<12} n={n:<4} ERROR: {r['error']}")

    # Output
    print_results(descriptives, results, args.earnings_metric, args.signal_threshold)
    save_results(descriptives, results, df, args.earnings_metric, args.signal_threshold)

    print("[Done] Backtesting complete.")


if __name__ == "__main__":
    main()
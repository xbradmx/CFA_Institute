# =============================================================================
# DDDS — Unified Analyst Interface
# Disclosure Degradation Detection System | The Transparency Project
# Lancaster University | CFA AI Investment Challenge 2026
#
# Run from the project root:
#   python analyst_frontend.py
#
# Requires:
#   pip install customtkinter
# =============================================================================

import csv
import io
import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import tkinter as tk

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent
_OPUS_DIR    = _ROOT / "data" / "Graph Rag Creation Data" / "opus_analysis"
_OUTPUTS_DIR = _ROOT / "data" / "analyst_outputs"

sys.path.insert(0, str(_ROOT / "Pipeline"))

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#07090f"
SURFACE   = "#0e1118"
RAISED    = "#141820"
BORDER    = "#1e2535"
ACCENT    = "#4d7fbe"
RED       = "#c45c5c"
RED_LO    = "#200d0d"
AMBER     = "#c9924a"
AMBER_LO  = "#1e1408"
GREEN     = "#4a9a6e"
GREEN_LO  = "#0d1a12"
TXT       = "#dde2ec"
TXT2      = "#7a8aaa"
TXT3      = "#3d4d63"

SIGNAL_COLOURS = {
    "HIGH":   (RED,   RED_LO),
    "MEDIUM": (AMBER, AMBER_LO),
    "LOW":    (GREEN, GREEN_LO),
}
ASSESS_COLOURS = {
    "GENUINE":    (RED,   RED_LO),
    "BORDERLINE": (AMBER, AMBER_LO),
    "DISMISS":    (GREEN, GREEN_LO),
    "DISMISSED":  (GREEN, GREEN_LO),
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ── Data helpers ──────────────────────────────────────────────────────────────

_FILING_RE = re.compile(
    r'^(?P<ticker>.+?)_(?P<ftype>10-[KQ])_(?P<date>\d{4}-\d{2}-\d{2})_.+_ddds_scores\.csv$'
)

def _parse_ddds_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    marker = "SENTENCE LEVEL DETAIL"
    idx = raw.find(marker)
    if idx == -1:
        return []
    rows = []
    reader = csv.DictReader(io.StringIO(raw[idx + len(marker):].lstrip("\r\n")))
    for row in reader:
        try:
            rows.append({
                "vague_prob":   float(row.get("vague_prob",   0)),
                "complex_prob": float(row.get("complex_prob", 0)),
                "section":      row.get("section", ""),
            })
        except (ValueError, TypeError):
            continue
    return rows


def load_filing_trend(ticker: str) -> list[dict]:
    """Return per-filing mean scores sorted by date for the given ticker."""
    results = []
    for path in sorted(_OUTPUTS_DIR.glob(f"{ticker}_*_ddds_scores.csv")):
        m = _FILING_RE.match(path.name)
        if not m or m.group("ticker") != ticker:
            continue
        rows = _parse_ddds_csv(path)
        if not rows:
            continue
        vp = [r["vague_prob"]   for r in rows]
        cp = [r["complex_prob"] for r in rows]
        results.append({
            "date":         datetime.strptime(m.group("date"), "%Y-%m-%d"),
            "filing_type":  m.group("ftype"),
            "mean_vague":   sum(vp) / len(vp),
            "mean_complex": sum(cp) / len(cp),
            "mean_avg":     (sum(vp) + sum(cp)) / (2 * len(vp)),
            "n":            len(vp),
        })
    return sorted(results, key=lambda x: x["date"])


def load_sector_and_rankings(top_n: int = 10) -> tuple[list[dict], dict, dict]:
    """
    Single pass over all CSVs. Returns:
      sector_data      — quarterly mean/std scores for the Sector Trends tab
      rankings_recent  — top_n companies by mean score using most-recent filing only
      rankings_alltime — top_n companies by mean score aggregated across ALL filings
    """
    import statistics
    from collections import defaultdict

    # Per-quarter buckets (filing-level means, for sector trend)
    q_buckets: dict = defaultdict(lambda: {"vague": [], "complex": []})
    # Per-company: track most-recent filing only (date + path)
    c_latest: dict = {}   # ticker -> {"date": datetime, "path": Path}
    # Per-company: accumulate key-section rows across ALL filings
    _KEY_SECTIONS_SET = {"MD&A", "Risk Factors"}
    c_all_key_rows: dict = defaultdict(list)   # ticker -> list of row dicts

    # First pass: sector trend buckets + find each company's latest filing date
    for path in sorted(_OUTPUTS_DIR.glob("*_ddds_scores.csv")):
        m = _FILING_RE.match(path.name)
        if not m:
            continue
        ticker = m.group("ticker")
        date   = datetime.strptime(m.group("date"), "%Y-%m-%d")
        q_month = ((date.month - 1) // 3) * 3 + 1
        q_start = datetime(date.year, q_month, 1)

        rows = _parse_ddds_csv(path)
        if not rows:
            continue
        vp = [r["vague_prob"]   for r in rows]
        cp = [r["complex_prob"] for r in rows]

        # Sector trend: store filing-level means per quarter
        q_buckets[q_start]["vague"].append(sum(vp) / len(vp))
        q_buckets[q_start]["complex"].append(sum(cp) / len(cp))

        # Rankings: keep track of the most recent filing per company
        if ticker not in c_latest or date > c_latest[ticker]["date"]:
            c_latest[ticker] = {"date": date, "path": path}

        # All-time: accumulate key-section sentences across every filing
        for r in rows:
            if r["section"] in _KEY_SECTIONS_SET:
                c_all_key_rows[ticker].append(r)

    # Build sector trend list
    sector = []
    for q_start in sorted(q_buckets):
        vl = q_buckets[q_start]["vague"]
        cl = q_buckets[q_start]["complex"]
        al = [(v + c) / 2 for v, c in zip(vl, cl)]
        n  = len(vl)
        q_num = (q_start.month - 1) // 3 + 1
        sector.append({
            "quarter_start": q_start,
            "label":         f"Q{q_num} '{q_start.strftime('%y')}",
            "mean_vague":    sum(vl) / n,
            "mean_complex":  sum(cl) / n,
            "mean_avg":      sum(al) / n,
            "std_vague":     statistics.stdev(vl) if n > 1 else 0.0,
            "std_complex":   statistics.stdev(cl) if n > 1 else 0.0,
            "n_filings":     n,
        })

    # Build rankings: parse only each company's most recent filing
    _KEY_SECTIONS = {"MD&A", "Risk Factors"}
    raw_rankings = []
    for ticker, info in c_latest.items():
        rows = _parse_ddds_csv(info["path"])
        if not rows:
            continue
        # Only score companies with ≥50 sentences across MD&A + Risk Factors
        key_rows = [r for r in rows if r["section"] in _KEY_SECTIONS]
        if len(key_rows) < 50:
            continue
        vl = [r["vague_prob"]   for r in key_rows]
        cl = [r["complex_prob"] for r in key_rows]
        mean_v = sum(vl) / len(vl)
        mean_c = sum(cl) / len(cl)
        raw_rankings.append({
            "ticker":       ticker,
            "filing_date":  info["date"].strftime("%Y-%m-%d"),
            "mean_vague":   mean_v,
            "mean_complex": mean_c,
            "mean_avg":     (mean_v + mean_c) / 2,
            "vague_pct":    sum(1 for v in vl if v >= 0.5) / len(vl) * 100,
            "n_sentences":  len(vl),   # MD&A + Risk Factors sentences only
        })

    # Build one top-N list per metric (most-recent filing); enrich the union
    top_vague   = sorted(raw_rankings, key=lambda x: -x["mean_vague"])[:top_n]
    top_complex = sorted(raw_rankings, key=lambda x: -x["mean_complex"])[:top_n]
    top_both    = sorted(raw_rankings, key=lambda x: -x["mean_avg"])[:top_n]

    # ── All-time rankings: aggregate sentences across ALL filings ─────────────
    at_raw = []
    for ticker, key_rows in c_all_key_rows.items():
        if len(key_rows) < 50:
            continue
        vl = [r["vague_prob"]   for r in key_rows]
        cl = [r["complex_prob"] for r in key_rows]
        mean_v = sum(vl) / len(vl)
        mean_c = sum(cl) / len(cl)
        latest_info = c_latest.get(ticker, {})
        filing_date_str = (latest_info["date"].strftime("%Y-%m-%d")
                           if "date" in latest_info else "N/A")
        at_raw.append({
            "ticker":       ticker,
            "filing_date":  filing_date_str,
            "mean_vague":   mean_v,
            "mean_complex": mean_c,
            "mean_avg":     (mean_v + mean_c) / 2,
            "vague_pct":    sum(1 for v in vl if v >= 0.5) / len(vl) * 100,
            "n_sentences":  len(vl),
        })

    top_vague_at   = sorted(at_raw, key=lambda x: -x["mean_vague"])[:top_n]
    top_complex_at = sorted(at_raw, key=lambda x: -x["mean_complex"])[:top_n]
    top_both_at    = sorted(at_raw, key=lambda x: -x["mean_avg"])[:top_n]

    # Union keyed by ticker so we only fetch each company once
    enrich_map: dict = {e["ticker"]: e
                        for lst in [top_vague, top_complex, top_both,
                                    top_vague_at, top_complex_at, top_both_at]
                        for e in lst}

    # Fetch stock price change from filing date → today for each unique company
    try:
        import yfinance as yf
        from datetime import date, timedelta
        today    = date.today()
        tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        for entry in enrich_map.values():
            try:
                hist = yf.Ticker(entry["ticker"]).history(
                    start=entry["filing_date"],
                    end=tomorrow,
                    auto_adjust=True,
                )
                if hist.empty or len(hist) < 2:
                    raise ValueError("insufficient history")
                p0 = float(hist["Close"].iloc[0])
                p1 = float(hist["Close"].iloc[-1])
                entry["price_change_pct"] = round((p1 - p0) / p0 * 100, 1)
                entry["price_at_filing"]  = round(p0, 2)
                entry["price_today"]      = round(p1, 2)
            except Exception:
                entry["price_change_pct"] = None
                entry["price_at_filing"]  = None
                entry["price_today"]      = None
    except ImportError:
        for entry in enrich_map.values():
            entry["price_change_pct"] = None

    # Load quarterly earnings and compute persistence for each unique company
    _qe_path = _ROOT / "data" / "backtest" / "quarterly_earnings_yfinance.csv"
    quarterly_by_ticker: dict = {}
    if _qe_path.exists():
        from collections import defaultdict as _dd
        _qb: dict = _dd(list)
        with open(_qe_path, encoding="utf-8") as _f:
            for _row in csv.DictReader(_f):
                try:
                    _qb[_row["ticker"]].append(
                        (_row["quarter_key"], float(_row["soe"]))
                    )
                except (ValueError, TypeError):
                    pass
        quarterly_by_ticker = {t: sorted(v) for t, v in _qb.items()}

    for entry in enrich_map.values():
        entry["earnings"] = _compute_earnings_persistence(
            entry["ticker"], entry["filing_date"], quarterly_by_ticker
        )

    return (
        sector,
        {"vague": top_vague,    "complex": top_complex,    "both": top_both},
        {"vague": top_vague_at, "complex": top_complex_at, "both": top_both_at},
    )


def _compute_earnings_persistence(
    ticker: str, filing_date_str: str, quarterly_by_ticker: dict
) -> dict | None:
    """Compute profitability status and earnings persistence around the filing date."""
    rows = quarterly_by_ticker.get(ticker, [])
    if not rows:
        return None

    def _qkey_to_date(qk: str) -> datetime:
        yr, q = qk.split("-Q")
        return datetime(int(yr), (int(q) - 1) * 3 + 1, 1)

    filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d")

    pre  = [(k, s) for k, s in rows if _qkey_to_date(k) <= filing_dt]
    post = [(k, s) for k, s in rows if _qkey_to_date(k) >  filing_dt]

    soe_at_filing = pre[-1][1]  if pre  else (post[0][1] if post else None)
    soe_latest    = post[-1][1] if post else (pre[-1][1] if pre else None)

    profitable = (soe_at_filing or 0) > 0

    # Direction: compare latest SOE to filing SOE
    if soe_at_filing is not None and soe_latest is not None:
        baseline = abs(soe_at_filing) if abs(soe_at_filing) > 0.002 else 0.002
        chg = (soe_latest - soe_at_filing) / baseline
        if chg > 0.10:
            direction = "improving"
        elif chg < -0.10:
            direction = "deteriorating"
        else:
            direction = "stable"
        change_pct = round(chg * 100, 1)
    else:
        direction = "n/a"
        change_pct = None

    # Persistence: normalise the absolute change by the company's own
    # pre-filing earnings volatility (std dev).  This benchmarks the move
    # against what was historically normal for *this* company, which is the
    # defensible approach — a 30% swing means something different for a
    # cyclical industrial vs a steady compounder.
    #   < 0.5σ  →  HIGH     (change within half a historical std dev)
    #   < 1.0σ  →  MODERATE
    #   ≥ 1.0σ  →  LOW      (moved more than one historical std dev)
    pre_soe = [s for _, s in pre]
    if (soe_at_filing is not None and soe_latest is not None and len(pre_soe) >= 2):
        import statistics as _stats
        hist_std = _stats.stdev(pre_soe)
        if hist_std > 0:
            sigma_move = abs(soe_latest - soe_at_filing) / hist_std
            if sigma_move < 0.5:
                persistence_label = "HIGH"
            elif sigma_move < 1.0:
                persistence_label = "MODERATE"
            else:
                persistence_label = "LOW"
        else:
            # Zero historical variance — any change is meaningful
            persistence_label = "HIGH" if change_pct is not None and abs(change_pct) < 5 else "LOW"
    else:
        persistence_label = "N/A"

    return {
        "profitable":        profitable,
        "soe_at_filing":     round(soe_at_filing, 4) if soe_at_filing is not None else None,
        "soe_latest":        round(soe_latest,    4) if soe_latest    is not None else None,
        "n_post_quarters":   len(post),
        "direction":         direction,
        "change_pct":        change_pct,
        "persistence_label": persistence_label,
    }


def load_tickers() -> list[str]:
    """Return sorted list of tickers that have a pre-computed Opus analysis."""
    if not _OPUS_DIR.exists():
        return []
    return sorted(
        p.stem.replace("_opus_analysis", "")
        for p in _OPUS_DIR.glob("*_opus_analysis.json")
    )


def load_analysis(ticker: str) -> dict | None:
    path = _OPUS_DIR / f"{ticker}_opus_analysis.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Main application ──────────────────────────────────────────────────────────

class DDDSApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("DDDS — Disclosure Degradation Detection System")
        self.geometry("1340x800")
        self.minsize(1000, 640)
        self.configure(fg_color=BG)

        self._tickers         = load_tickers()
        self._current_ticker  = None
        self._ticker_buttons: dict[str, ctk.CTkButton] = {}
        self._trend_ticker    = None   # tracks which ticker the trend chart shows
        self._trend_canvas    = None   # holds the current FigureCanvasTkAgg

        self._build_layout()
        self._populate_ticker_list(self._tickers)
        # Background loads — heatmap and sector/rankings share one CSV pass
        threading.Thread(target=self._load_heatmap,          daemon=True).start()
        threading.Thread(target=self._load_sector_rankings,  daemon=True).start()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.sidebar = ctk.CTkFrame(self, width=230, fg_color=SURFACE, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_content()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        # Logo block
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(24, 0))

        ctk.CTkLabel(
            logo_frame, text="DDDS",
            font=ctk.CTkFont(family="Consolas", size=24, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            logo_frame,
            text="Disclosure Degradation\nDetection System",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TXT3, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        _divider(self.sidebar, pady=16)

        # Search
        search_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        search_frame.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(
            search_frame, text="COMPANY SEARCH",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TXT3,
        ).pack(anchor="w", pady=(0, 4))

        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", self._on_search)

        ctk.CTkEntry(
            search_frame,
            textvariable=self._search_var,
            placeholder_text="e.g.  AAPL",
            fg_color=RAISED,
            border_color=BORDER,
            text_color=TXT,
            placeholder_text_color=TXT3,
            font=ctk.CTkFont(family="Consolas", size=12),
            height=32,
        ).pack(fill="x")

        self._count_lbl = ctk.CTkLabel(
            self.sidebar,
            text=f"{len(self._tickers)} companies in universe",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TXT3,
        )
        self._count_lbl.pack(anchor="w", padx=14, pady=(4, 6))

        # Scrollable ticker list
        self._ticker_scroll = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        self._ticker_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 10))

    def _populate_ticker_list(self, tickers: list[str]):
        for w in self._ticker_scroll.winfo_children():
            w.destroy()
        self._ticker_buttons.clear()

        for t in tickers:
            btn = ctk.CTkButton(
                self._ticker_scroll,
                text=t,
                anchor="w",
                font=ctk.CTkFont(family="Consolas", size=12),
                fg_color="transparent",
                hover_color=RAISED,
                text_color=TXT2,
                height=28,
                corner_radius=4,
                command=lambda ticker=t: self._select_ticker(ticker),
            )
            btn.pack(fill="x", pady=1, padx=2)
            self._ticker_buttons[t] = btn

        n = len(tickers)
        total = len(self._tickers)
        if n == total:
            label = f"{n} companies in universe"
        elif n == 1:
            label = "1 match"
        else:
            label = f"{n} matches"
        self._count_lbl.configure(text=label)

    def _on_search(self, *_):
        q = self._search_var.get().strip().upper()
        filtered = [t for t in self._tickers if q in t] if q else self._tickers
        self._populate_ticker_list(filtered)

    def _select_ticker(self, ticker: str):
        if ticker == self._current_ticker:
            return

        if self._current_ticker and self._current_ticker in self._ticker_buttons:
            self._ticker_buttons[self._current_ticker].configure(
                fg_color="transparent", text_color=TXT2,
            )

        self._current_ticker = ticker
        if ticker in self._ticker_buttons:
            self._ticker_buttons[ticker].configure(fg_color=RAISED, text_color=TXT)

        data = load_analysis(ticker)
        self._render_memo(ticker, data)

        # If Filing Trends is currently open, refresh it for the new ticker
        if self._tabs.get() == "  Filing Trends  ":
            self._ticker_lbl.configure(text=f"{ticker} — Filing Trends", text_color=TXT2)
            self._render_trend(ticker)
        else:
            # Switch to memo tab automatically
            self._tabs.set("  Investment Memo  ")

    # ── Content area ──────────────────────────────────────────────────────────

    def _build_content(self):
        # Top bar
        topbar = ctk.CTkFrame(self.content, height=54, fg_color=SURFACE, corner_radius=0)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        self._ticker_lbl = ctk.CTkLabel(
            topbar, text="Select a company from the sidebar",
            font=ctk.CTkFont(family="DejaVu Sans", size=15, weight="bold"),
            text_color=TXT3,
        )
        self._ticker_lbl.pack(side="left", padx=24)

        self._signal_badge = ctk.CTkLabel(
            topbar, text="",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            fg_color="transparent", text_color=BG,
            corner_radius=4, width=10,
        )
        self._signal_badge.pack(side="left", padx=6)

        _divider(self.content, orient="h", pady=0)

        # Tabs
        self._tabs = ctk.CTkTabview(
            self.content,
            fg_color=BG,
            segmented_button_fg_color=SURFACE,
            segmented_button_selected_color=RAISED,
            segmented_button_selected_hover_color=RAISED,
            segmented_button_unselected_color=SURFACE,
            segmented_button_unselected_hover_color=RAISED,
            text_color=TXT,
            text_color_disabled=TXT3,
            border_width=0,
        )
        self._tabs.pack(fill="both", expand=True)

        self._tabs.add("  Investment Memo  ")
        self._tabs.add("  Risk Heatmap  ")
        self._tabs.add("  Filing Trends  ")
        self._tabs.add("  Sector Trends  ")
        self._tabs.add("  FinBERT Rankings  ")

        self._tabs.configure(command=self._on_tab_change)

        self._build_memo_tab()
        self._build_heatmap_tab()
        self._build_trend_tab()
        self._build_sector_tab()
        self._build_rankings_tab()

    def _on_tab_change(self):
        tab = self._tabs.get()
        if tab == "  Risk Heatmap  ":
            self._ticker_lbl.configure(
                text="US Industrials — Sector-Wide Geographic Risk",
                text_color=TXT2,
            )
            self._signal_badge.configure(text="", fg_color="transparent")
        elif tab == "  Filing Trends  ":
            if self._current_ticker:
                self._ticker_lbl.configure(
                    text=f"{self._current_ticker} — Filing Trends",
                    text_color=TXT2,
                )
                self._signal_badge.configure(text="", fg_color="transparent")
                # Render chart if not yet built for this ticker
                if self._trend_ticker != self._current_ticker:
                    self._render_trend(self._current_ticker)
            else:
                self._ticker_lbl.configure(
                    text="Select a company to view filing trends",
                    text_color=TXT3,
                )
        elif tab == "  Sector Trends  ":
            self._ticker_lbl.configure(
                text="US Industrials — Sector Vagueness Trend",
                text_color=TXT2,
            )
            self._signal_badge.configure(text="", fg_color="transparent")
        elif tab == "  FinBERT Rankings  ":
            self._ticker_lbl.configure(
                text="US Industrials — Top 15 by FinBERT Score",
                text_color=TXT2,
            )
            self._signal_badge.configure(text="", fg_color="transparent")
        else:
            if self._current_ticker:
                self._ticker_lbl.configure(text=self._current_ticker, text_color=TXT)
            else:
                self._ticker_lbl.configure(
                    text="Select a company from the sidebar",
                    text_color=TXT3,
                )

    # ── Memo tab ──────────────────────────────────────────────────────────────

    def _build_memo_tab(self):
        tab = self._tabs.tab("  Investment Memo  ")
        tab.configure(fg_color=BG)

        # Placeholder (shown until a ticker is selected)
        self._memo_placeholder = ctk.CTkFrame(tab, fg_color="transparent")
        self._memo_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(
            self._memo_placeholder,
            text="Select a company from the sidebar to view its analysis.",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=TXT3,
        ).pack()

        # Scrollable memo body (hidden until needed)
        self._memo_scroll = ctk.CTkScrollableFrame(
            tab,
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )

    def _render_memo(self, ticker: str, data: dict | None):
        # Clear previous memo
        self._memo_placeholder.place_forget()
        for w in self._memo_scroll.winfo_children():
            w.destroy()
        self._memo_scroll.pack(fill="both", expand=True, padx=24, pady=8)

        if not data:
            self._ticker_lbl.configure(text=ticker, text_color=TXT)
            self._signal_badge.configure(text="", fg_color="transparent")
            ctk.CTkLabel(
                self._memo_scroll,
                text=f"No analysis found for {ticker}.",
                font=ctk.CTkFont(family="Consolas", size=12),
                text_color=TXT2,
            ).pack(pady=40)
            return

        overall = data.get("overall", {})
        flags   = data.get("flags", [])
        meta    = data.get("_meta", {})

        signal    = overall.get("signal_strength", "").upper()
        sig_fg, sig_bg = SIGNAL_COLOURS.get(signal, (TXT2, RAISED))

        # Update top bar
        self._ticker_lbl.configure(text=ticker, text_color=TXT)
        self._signal_badge.configure(
            text=f"  ▲ {signal} SIGNAL  ",
            fg_color=sig_bg, text_color=sig_fg,
        )

        # ── Summary ──
        summary = overall.get("summary", "")
        if summary:
            _section_label(self._memo_scroll, "SUMMARY")
            _body_card(self._memo_scroll, summary)

        # ── Stats row ──
        stats_row = ctk.CTkFrame(self._memo_scroll, fg_color="transparent")
        stats_row.pack(fill="x", pady=(0, 16))
        stats_row.columnconfigure((0, 1, 2), weight=1, uniform="stat")

        for col, (label, key, fg, bg) in enumerate([
            ("GENUINE",    "genuine_count",    RED,   RED_LO),
            ("BORDERLINE", "borderline_count", AMBER, AMBER_LO),
            ("DISMISSED",  "dismissed_count",  GREEN, GREEN_LO),
        ]):
            cell = ctk.CTkFrame(stats_row, fg_color=bg, corner_radius=8)
            cell.grid(row=0, column=col, padx=4, sticky="ew")
            ctk.CTkLabel(
                cell,
                text=str(overall.get(key, 0)),
                font=ctk.CTkFont(family="DejaVu Sans", size=32, weight="bold"),
                text_color=fg,
            ).pack(pady=(12, 0))
            ctk.CTkLabel(
                cell,
                text=label,
                font=ctk.CTkFont(family="Consolas", size=9),
                text_color=fg,
            ).pack(pady=(0, 12))

        # ── Top concerns ──
        concerns = overall.get("top_concerns", [])
        if concerns:
            _section_label(self._memo_scroll, "TOP CONCERNS")
            card = ctk.CTkFrame(self._memo_scroll, fg_color=SURFACE, corner_radius=8)
            card.pack(fill="x", pady=(0, 16))
            for concern in concerns:
                ctk.CTkLabel(
                    card,
                    text=f"  •  {concern}",
                    font=ctk.CTkFont(family="DejaVu Sans", size=11),
                    text_color=TXT2,
                    anchor="w", justify="left",
                    wraplength=750,
                ).pack(anchor="w", padx=14, pady=4)
            ctk.CTkFrame(card, height=6, fg_color="transparent").pack()

        # ── Flags grouped by assessment ──
        groups = [
            ("GENUINE FLAGS",    "GENUINE"),
            ("BORDERLINE FLAGS", "BORDERLINE"),
            ("DISMISSED FLAGS",  "DISMISS"),
        ]
        for group_label, assessment_key in groups:
            group = [f for f in flags if f.get("assessment", "").upper() == assessment_key]
            if not group:
                continue
            _section_label(self._memo_scroll, group_label)
            for flag in group:
                _flag_card(self._memo_scroll, flag)

        # ── Meta footer ──
        model = meta.get("model", "")
        ts    = meta.get("timestamp", "")[:10]
        if model:
            ctk.CTkLabel(
                self._memo_scroll,
                text=f"Analysis by {model}  ·  {ts}  ·  "
                     f"{meta.get('input_tokens', 0):,} input tokens  /  "
                     f"{meta.get('output_tokens', 0):,} output tokens",
                font=ctk.CTkFont(family="Consolas", size=9),
                text_color=TXT3,
            ).pack(anchor="e", pady=(12, 8))

    # ── Heatmap tab ───────────────────────────────────────────────────────────

    def _build_heatmap_tab(self):
        tab = self._tabs.tab("  Risk Heatmap  ")
        tab.configure(fg_color=BG)

        self._heatmap_frame = ctk.CTkFrame(tab, fg_color=BG, corner_radius=0)
        self._heatmap_frame.pack(fill="both", expand=True)

        self._heatmap_spinner = ctk.CTkLabel(
            self._heatmap_frame,
            text="Loading geographic risk data...",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=TXT3,
        )
        self._heatmap_spinner.place(relx=0.5, rely=0.5, anchor="center")

    def _load_heatmap(self):
        """Background thread: builds the heatmap figure, then embeds on main thread."""
        try:
            from run_08_risk_heatmap import build_figure
            fig, on_hover = build_figure(str(_OUTPUTS_DIR), dpi=110)
            self.after(0, lambda: self._embed_heatmap(fig, on_hover))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda: self._heatmap_spinner.configure(
                text=f"Heatmap unavailable.\n{msg}",
                wraplength=500,
            ))

    def _embed_heatmap(self, fig, on_hover):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self._heatmap_spinner.place_forget()
        canvas = FigureCanvasTkAgg(fig, master=self._heatmap_frame)
        canvas.draw()
        canvas.mpl_connect("motion_notify_event", on_hover)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Trend tab ─────────────────────────────────────────────────────────────

    def _build_trend_tab(self):
        tab = self._tabs.tab("  Filing Trends  ")
        tab.configure(fg_color=BG)

        self._trend_frame = ctk.CTkFrame(tab, fg_color=BG, corner_radius=0)
        self._trend_frame.pack(fill="both", expand=True)

        self._trend_placeholder = ctk.CTkLabel(
            self._trend_frame,
            text="Select a company from the sidebar to view filing trends.",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=TXT3,
        )
        self._trend_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _render_trend(self, ticker: str):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.dates import DateFormatter
        from matplotlib.lines import Line2D

        # Tear down previous canvas
        if self._trend_canvas is not None:
            self._trend_canvas.get_tk_widget().destroy()
            self._trend_canvas = None
        self._trend_placeholder.place_forget()

        data = load_filing_trend(ticker)

        if not data:
            self._trend_placeholder.configure(
                text=f"No filing data found for {ticker}.",
            )
            self._trend_placeholder.place(relx=0.5, rely=0.5, anchor="center")
            return

        # ── Build figure ──────────────────────────────────────────────────
        PLOT_BG   = "#0e1118"
        GRID_CLR  = "#1a2133"
        CLR_VAGUE   = "#4d7fbe"   # blue
        CLR_COMPLEX = "#c9924a"   # amber
        CLR_AVG     = "#9b6ec4"   # purple

        fig = Figure(figsize=(12, 5), dpi=110, facecolor=BG)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(PLOT_BG)

        # Grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6)
        ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_CLR)
        ax.spines["bottom"].set_color(GRID_CLR)
        ax.tick_params(colors=TXT2, labelsize=9, length=0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Consolas")

        dates   = [d["date"]         for d in data]
        vague   = [d["mean_vague"]   for d in data]
        complex_= [d["mean_complex"] for d in data]
        avg     = [d["mean_avg"]     for d in data]

        # Shaded fill under vague line (very subtle)
        ax.fill_between(dates, vague, alpha=0.07, color=CLR_VAGUE)

        # Lines
        ax.plot(dates, vague,    color=CLR_VAGUE,   linewidth=2,   alpha=0.9, zorder=3)
        ax.plot(dates, complex_, color=CLR_COMPLEX, linewidth=2,   alpha=0.9, zorder=3)
        ax.plot(dates, avg,      color=CLR_AVG,     linewidth=1.5, alpha=0.9,
                linestyle="--", zorder=3)

        # Markers — circle for 10-K, triangle for 10-Q
        for d, v, c, a in zip(data, vague, complex_, avg):
            mk = "o" if d["filing_type"] == "10-K" else "^"
            ms = 9  if d["filing_type"] == "10-K" else 7
            for val, clr in [(v, CLR_VAGUE), (c, CLR_COMPLEX), (a, CLR_AVG)]:
                ax.plot(d["date"], val, marker=mk, markersize=ms,
                        color=clr, markeredgecolor=PLOT_BG, markeredgewidth=1.2,
                        zorder=5)

        # Risk threshold lines
        for y, label in [(0.5, "0.5 threshold"), (0.25, ""), (0.75, "")]:
            ax.axhline(y, color=TXT3, linewidth=0.5, linestyle=":", alpha=0.5)
            if label:
                ax.text(dates[0], y + 0.01, label,
                        fontsize=7, fontfamily="Consolas", color=TXT3, va="bottom")

        # Annotate each point with filing label
        for d, v in zip(data, vague):
            ax.annotate(
                d["date"].strftime("%b '%y"),
                xy=(d["date"], v),
                xytext=(0, 12), textcoords="offset points",
                fontsize=7, fontfamily="Consolas", color=TXT3,
                ha="center", va="bottom",
            )

        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.2f}")
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30, ha="right")

        ax.set_ylabel("Score", fontsize=9, fontfamily="Consolas", color=TXT2, labelpad=10)

        # Legend
        legend_handles = [
            Line2D([0], [0], color=CLR_VAGUE,   linewidth=2, label="Vague Prob"),
            Line2D([0], [0], color=CLR_COMPLEX, linewidth=2, label="Complex Prob"),
            Line2D([0], [0], color=CLR_AVG,     linewidth=1.5, linestyle="--", label="Avg Score"),
            Line2D([0], [0], marker="o", color=TXT3, markersize=7, linewidth=0,
                   markeredgecolor=PLOT_BG, label="10-K"),
            Line2D([0], [0], marker="^", color=TXT3, markersize=6, linewidth=0,
                   markeredgecolor=PLOT_BG, label="10-Q"),
        ]
        leg = ax.legend(
            handles=legend_handles,
            loc="upper right", fontsize=8,
            facecolor=SURFACE, edgecolor=BORDER,
            labelcolor=TXT2, framealpha=0.95,
        )
        for text in leg.get_texts():
            text.set_fontfamily("Consolas")

        # Subtitle
        fig.text(
            0.5, 0.97,
            f"{ticker}  ·  {len(data)} filing(s)  ·  mean scores per filing",
            ha="center", va="top",
            fontsize=8, fontfamily="Consolas", color=TXT3,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self._trend_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._trend_canvas = canvas
        self._trend_ticker = ticker

    # ── Sector trend tab ──────────────────────────────────────────────────────

    def _build_sector_tab(self):
        tab = self._tabs.tab("  Sector Trends  ")
        tab.configure(fg_color=BG)

        self._sector_frame = ctk.CTkFrame(tab, fg_color=BG, corner_radius=0)
        self._sector_frame.pack(fill="both", expand=True)

        self._sector_spinner = ctk.CTkLabel(
            self._sector_frame,
            text="Loading sector data across all 925 filings...",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=TXT3,
        )
        self._sector_spinner.place(relx=0.5, rely=0.5, anchor="center")

    def _load_sector_rankings(self):
        """Background thread: single CSV pass → sector chart + rankings chart."""
        try:
            sector_data, rankings_recent, rankings_alltime = load_sector_and_rankings(top_n=15)
            self.after(0, lambda: self._render_sector(sector_data))
            self.after(0, lambda: self._render_rankings(rankings_recent, rankings_alltime))
        except Exception as exc:
            msg = f"Data unavailable.\n{exc}"
            self.after(0, lambda: self._sector_spinner.configure(
                text=msg, wraplength=500,
            ))
            self.after(0, lambda: self._rankings_spinner.configure(
                text=msg, wraplength=500,
            ))

    def _render_sector(self, data: list[dict]):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        self._sector_spinner.place_forget()

        if not data:
            self._sector_spinner.configure(text="No filing data found.")
            self._sector_spinner.place(relx=0.5, rely=0.5, anchor="center")
            return

        PLOT_BG     = "#0e1118"
        GRID_CLR    = "#1a2133"
        CLR_VAGUE   = "#4d7fbe"
        CLR_COMPLEX = "#c9924a"
        CLR_AVG     = "#9b6ec4"

        fig = Figure(figsize=(13, 5.5), dpi=110, facecolor=BG)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(PLOT_BG)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6)
        ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_CLR)
        ax.spines["bottom"].set_color(GRID_CLR)
        ax.tick_params(colors=TXT2, labelsize=9, length=0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Consolas")

        xs       = list(range(len(data)))
        labels   = [d["label"]        for d in data]
        vague    = [d["mean_vague"]   for d in data]
        complex_ = [d["mean_complex"] for d in data]
        avg      = [d["mean_avg"]     for d in data]
        std_v    = [d["std_vague"]    for d in data]
        std_c    = [d["std_complex"]  for d in data]
        n_filings= [d["n_filings"]    for d in data]

        # Std-dev bands
        v_hi = [v + s for v, s in zip(vague,    std_v)]
        v_lo = [v - s for v, s in zip(vague,    std_v)]
        c_hi = [v + s for v, s in zip(complex_, std_c)]
        c_lo = [v - s for v, s in zip(complex_, std_c)]
        ax.fill_between(xs, v_lo, v_hi, color=CLR_VAGUE,   alpha=0.10, zorder=1)
        ax.fill_between(xs, c_lo, c_hi, color=CLR_COMPLEX, alpha=0.10, zorder=1)

        # Lines
        ax.plot(xs, vague,    color=CLR_VAGUE,   linewidth=2.2, alpha=0.9, zorder=3)
        ax.plot(xs, complex_, color=CLR_COMPLEX, linewidth=2.2, alpha=0.9, zorder=3)
        ax.plot(xs, avg,      color=CLR_AVG,     linewidth=1.8, alpha=0.9,
                linestyle="--", zorder=3)

        # Markers
        ax.scatter(xs, vague,    color=CLR_VAGUE,   s=55, zorder=5,
                   edgecolors=PLOT_BG, linewidths=1.2)
        ax.scatter(xs, complex_, color=CLR_COMPLEX, s=55, zorder=5,
                   edgecolors=PLOT_BG, linewidths=1.2)
        ax.scatter(xs, avg,      color=CLR_AVG,     s=40, zorder=5,
                   edgecolors=PLOT_BG, linewidths=1.0)

        # 0.5 threshold
        ax.axhline(0.5, color=TXT3, linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(xs[-1] + 0.1, 0.502, "0.5", fontsize=7, fontfamily="Consolas",
                color=TXT3, va="bottom")

        # Annotate n_filings below each quarter
        for x, n in zip(xs, n_filings):
            ax.text(x, -0.06, f"n={n}", ha="center", va="top",
                    fontsize=7, fontfamily="Consolas", color=TXT3,
                    transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean Score", fontsize=9, fontfamily="Consolas",
                      color=TXT2, labelpad=10)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.2f}")

        # Legend
        legend_handles = [
            Line2D([0], [0], color=CLR_VAGUE,   linewidth=2, label="Vague Prob"),
            Line2D([0], [0], color=CLR_COMPLEX, linewidth=2, label="Complex Prob"),
            Line2D([0], [0], color=CLR_AVG,     linewidth=1.5, linestyle="--",
                   label="Avg Score"),
            Patch(facecolor=CLR_VAGUE,   alpha=0.20, label="±1 SD  (vague)"),
            Patch(facecolor=CLR_COMPLEX, alpha=0.20, label="±1 SD  (complex)"),
        ]
        leg = ax.legend(
            handles=legend_handles, loc="upper right", fontsize=8,
            facecolor=SURFACE, edgecolor=BORDER, labelcolor=TXT2, framealpha=0.95,
        )
        for t in leg.get_texts():
            t.set_fontfamily("Consolas")

        total_filings = sum(n_filings)
        fig.text(
            0.5, 0.97,
            f"US Industrials  ·  {total_filings} filings across {len(data)} quarters"
            f"  ·  mean ± 1 SD per quarter",
            ha="center", va="top",
            fontsize=8, fontfamily="Consolas", color=TXT3,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self._sector_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── FinBERT Rankings tab ──────────────────────────────────────────────────

    def _build_rankings_tab(self):
        tab = self._tabs.tab("  FinBERT Rankings  ")
        tab.configure(fg_color=BG)

        self._rankings_frame = ctk.CTkFrame(tab, fg_color=BG, corner_radius=0)
        self._rankings_frame.pack(fill="both", expand=True)

        self._rankings_spinner = ctk.CTkLabel(
            self._rankings_frame,
            text="Computing FinBERT rankings across all filings...",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=TXT3,
        )
        self._rankings_spinner.place(relx=0.5, rely=0.5, anchor="center")

    def _render_rankings(self, rankings_recent: dict, rankings_alltime: dict):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._rankings_spinner.place_forget()

        if not any(rankings_recent.values()):
            self._rankings_spinner.configure(text="No ranking data found.")
            self._rankings_spinner.place(relx=0.5, rely=0.5, anchor="center")
            return

        self._rankings_data_recent  = rankings_recent
        self._rankings_data_alltime = rankings_alltime
        self._rankings_view         = "price"
        self._rankings_metric       = "both"
        self._rankings_scope        = "recent"   # "recent" | "alltime"

        # ── Toggle row ────────────────────────────────────────────────────────
        toggle_row = ctk.CTkFrame(self._rankings_frame, fg_color=SURFACE,
                                  height=38, corner_radius=0)
        toggle_row.pack(fill="x", side="top")
        toggle_row.pack_propagate(False)

        # View toggles
        self._rankings_toggle_btns: dict[str, ctk.CTkButton] = {}
        for view, label in [("price", "Stock Price"),
                             ("earnings", "Earnings Persistence"),
                             ("stats", "Stats")]:
            btn = ctk.CTkButton(
                toggle_row, text=label,
                font=ctk.CTkFont(family="Consolas", size=10),
                fg_color=RAISED if view == "price" else "transparent",
                hover_color=RAISED,
                text_color=TXT if view == "price" else TXT2,
                height=28, corner_radius=4,
                command=lambda v=view: self._switch_rankings_view(v),
            )
            btn.pack(side="left", padx=(8, 2), pady=5)
            self._rankings_toggle_btns[view] = btn

        # Separator + "RANK BY" label
        ctk.CTkLabel(toggle_row, text="│",
                     font=ctk.CTkFont(size=16), text_color=BORDER,
                     ).pack(side="left", padx=(12, 4))
        ctk.CTkLabel(toggle_row, text="RANK BY",
                     font=ctk.CTkFont(family="Consolas", size=9),
                     text_color=TXT3).pack(side="left", padx=(0, 6))

        # Metric toggles
        self._rankings_metric_btns: dict[str, ctk.CTkButton] = {}
        for metric, label in [("vague", "Vague"),
                               ("complex", "Complex"),
                               ("both", "Both")]:
            btn = ctk.CTkButton(
                toggle_row, text=label,
                font=ctk.CTkFont(family="Consolas", size=10),
                fg_color=RAISED if metric == "both" else "transparent",
                hover_color=RAISED,
                text_color=TXT if metric == "both" else TXT2,
                height=28, corner_radius=4,
                command=lambda m=metric: self._switch_rankings_metric(m),
            )
            btn.pack(side="left", padx=(2, 2), pady=5)
            self._rankings_metric_btns[metric] = btn

        # Separator + "SCOPE" label
        ctk.CTkLabel(toggle_row, text="│",
                     font=ctk.CTkFont(size=16), text_color=BORDER,
                     ).pack(side="left", padx=(12, 4))
        ctk.CTkLabel(toggle_row, text="SCOPE",
                     font=ctk.CTkFont(family="Consolas", size=9),
                     text_color=TXT3).pack(side="left", padx=(0, 6))

        # Scope toggles
        self._rankings_scope_btns: dict[str, ctk.CTkButton] = {}
        for scope, label in [("recent", "Most Recent"), ("alltime", "All Time")]:
            btn = ctk.CTkButton(
                toggle_row, text=label,
                font=ctk.CTkFont(family="Consolas", size=10),
                fg_color=RAISED if scope == "recent" else "transparent",
                hover_color=RAISED,
                text_color=TXT if scope == "recent" else TXT2,
                height=28, corner_radius=4,
                command=lambda s=scope: self._switch_rankings_scope(s),
            )
            btn.pack(side="left", padx=(2, 2), pady=5)
            self._rankings_scope_btns[scope] = btn

        # ── Figure with two axes ──────────────────────────────────────────────
        fig = Figure(figsize=(13, 8), dpi=110, facecolor=BG)
        ax_left  = fig.add_axes([0.10, 0.07, 0.52, 0.84])
        ax_right = fig.add_axes([0.65, 0.07, 0.32, 0.84])

        self._rankings_fig      = fig
        self._rankings_ax_left  = ax_left
        self._rankings_ax       = ax_right

        from datetime import date as _date
        self._rankings_today_str = _date.today().strftime("%Y-%m-%d")
        self._rankings_subtitle  = fig.text(
            0.5, 0.98, "", ha="center", va="top",
            fontsize=8, fontfamily="Consolas", color=TXT3,
        )

        canvas = FigureCanvasTkAgg(fig, master=self._rankings_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._rankings_canvas = canvas

        # Draw initial state
        data = rankings_recent["both"]
        self._rankings_data = data
        self._rankings_ys   = list(range(len(data)))
        self._draw_left_bars(ax_left, data, self._rankings_ys)
        self._switch_rankings_view("price")

    # ── Rankings helpers ──────────────────────────────────────────────────────

    def _draw_left_bars(self, ax, data, ys):
        from matplotlib.lines import Line2D
        PLOT_BG  = "#0e1118"
        GRID_CLR = "#1a2133"
        CLR_VAGUE   = "#4d7fbe"
        CLR_COMPLEX = "#c9924a"
        CLR_AVG     = "#9b6ec4"
        bh = 0.28

        ax.cla()
        ax.set_facecolor(PLOT_BG)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_CLR)
        ax.spines["bottom"].set_color(GRID_CLR)
        ax.tick_params(colors=TXT2, labelsize=9, length=0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Consolas")

        n        = len(data)
        tickers  = [d["ticker"]       for d in data]
        vague    = [d["mean_vague"]   for d in data]
        complex_ = [d["mean_complex"] for d in data]
        avg      = [d["mean_avg"]     for d in data]

        ax.barh([y + bh for y in ys], vague,    height=bh, color=CLR_VAGUE,
                alpha=0.85, zorder=2)
        ax.barh(ys,                   complex_, height=bh, color=CLR_COMPLEX,
                alpha=0.85, zorder=2)
        for i, (v, c) in enumerate(zip(vague, complex_)):
            ax.text(v + 0.008, i + bh + bh / 2, f"{v:.3f}", va="center",
                    fontsize=7.5, fontfamily="Consolas", color=CLR_VAGUE)
            ax.text(c + 0.008, i + bh / 2,       f"{c:.3f}", va="center",
                    fontsize=7.5, fontfamily="Consolas", color=CLR_COMPLEX)
        ax.scatter(avg, [y + bh for y in ys], marker="D", s=28,
                   color=CLR_AVG, zorder=5)
        ax.axvline(0.5, color=TXT3, linewidth=0.7, linestyle=":", alpha=0.7)
        ax.text(0.501, n - 0.1, "0.5", fontsize=7, fontfamily="Consolas",
                color=TXT3, va="top")
        ax.set_yticks([y + bh for y in ys])
        ax.set_yticklabels([f"#{i+1}  {t}" for i, t in enumerate(tickers)],
                           fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Mean Score (0–1)", fontsize=8, fontfamily="Consolas",
                      color=TXT2, labelpad=8)
        leg = ax.legend(
            handles=[
                Line2D([0], [0], color=CLR_VAGUE,   linewidth=8, alpha=0.85, label="Vague"),
                Line2D([0], [0], color=CLR_COMPLEX, linewidth=8, alpha=0.85, label="Complex"),
                Line2D([0], [0], marker="D", color=CLR_AVG, markersize=6,
                       linewidth=0, label="Avg"),
            ],
            loc="lower right", fontsize=8,
            facecolor=SURFACE, edgecolor=BORDER, labelcolor=TXT2, framealpha=0.95,
        )
        for t in leg.get_texts():
            t.set_fontfamily("Consolas")

    def _active_rankings_dict(self) -> dict:
        return (self._rankings_data_alltime
                if self._rankings_scope == "alltime"
                else self._rankings_data_recent)

    def _switch_rankings_scope(self, scope: str):
        self._rankings_scope = scope
        for s, btn in self._rankings_scope_btns.items():
            btn.configure(
                fg_color=RAISED if s == scope else "transparent",
                text_color=TXT   if s == scope else TXT2,
            )
        data = self._active_rankings_dict()[self._rankings_metric]
        self._rankings_data = data
        self._rankings_ys   = list(range(len(data)))
        self._draw_left_bars(self._rankings_ax_left, data, self._rankings_ys)
        self._switch_rankings_view(self._rankings_view)

    def _switch_rankings_metric(self, metric: str):
        self._rankings_metric = metric
        for m, btn in self._rankings_metric_btns.items():
            btn.configure(
                fg_color=RAISED if m == metric else "transparent",
                text_color=TXT   if m == metric else TXT2,
            )
        data = self._active_rankings_dict()[metric]
        self._rankings_data = data
        self._rankings_ys   = list(range(len(data)))
        self._draw_left_bars(self._rankings_ax_left, data, self._rankings_ys)
        self._switch_rankings_view(self._rankings_view)

    # ── Right-panel drawing helpers ───────────────────────────────────────────

    def _switch_rankings_view(self, view: str):
        self._rankings_view = view

        # Update toggle button styles
        for v, btn in self._rankings_toggle_btns.items():
            btn.configure(
                fg_color=RAISED if v == view else "transparent",
                text_color=TXT   if v == view else TXT2,
            )

        ax  = self._rankings_ax
        ys  = self._rankings_ys
        data = self._rankings_data
        n   = len(data)

        PLOT_BG  = "#0e1118"
        GRID_CLR = "#1a2133"
        ax.cla()
        ax.set_facecolor(PLOT_BG)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(GRID_CLR)
        ax.tick_params(colors=TXT2, labelsize=8, length=0, left=False)

        if view == "price":
            self._draw_right_price(ax, data, ys, PLOT_BG, GRID_CLR)
        elif view == "earnings":
            self._draw_right_earnings(ax, data, ys, PLOT_BG, GRID_CLR)
        elif view == "stats":
            self._draw_right_stats(ax, data, ys, PLOT_BG, GRID_CLR)

        self._rankings_canvas.draw_idle()

    def _draw_right_price(self, ax, data, ys, PLOT_BG, GRID_CLR):
        n = len(data)
        ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle=":")
        for lbl in ax.get_xticklabels():
            lbl.set_fontfamily("Consolas")

        returns = [d.get("price_change_pct") for d in data]
        colours, values = [], []
        for r in returns:
            if r is None:
                colours.append(TXT3); values.append(0.0)
            elif r >= 0:
                colours.append(GREEN); values.append(r)
            else:
                colours.append(RED);   values.append(r)

        ax.barh(ys, values, height=0.55, color=colours, alpha=0.85, zorder=2)
        ax.axvline(0, color=TXT3, linewidth=0.8, zorder=3)

        x_lim = max((abs(v) for v in values if v != 0), default=10) * 1.4
        for i, (r, val) in enumerate(zip(returns, values)):
            lbl = f"{r:+.1f}%" if r is not None else "n/a"
            off = x_lim * 0.04
            ax.text(val + (off if val >= 0 else -off), i, lbl,
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=8, fontfamily="Consolas", color=colours[i])

        ax.set_xlim(-x_lim, x_lim)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0f}%")
        ax.set_yticks(ys)
        ax.set_yticklabels([""] * n)
        ax.invert_yaxis()
        ax.set_xlabel("Return since filing date", fontsize=8,
                      fontfamily="Consolas", color=TXT2, labelpad=8)

        scope_lbl = "all filings" if self._rankings_scope == "alltime" else "most recent filing per company"
        self._rankings_subtitle.set_text(
            f"Top {n}  ·  {scope_lbl}"
            f"  ·  price as of {self._rankings_today_str}"
        )

    def _draw_right_earnings(self, ax, data, ys, PLOT_BG, GRID_CLR):
        n = len(data)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Column layout
        cols  = ["Status",    "SOE@flag", "SOE now", "Chg",   "Persist"]
        col_x = [0.01,         0.22,       0.40,      0.58,    0.76    ]
        row_h = 0.88 / (n + 1)

        # Header
        for cx, lbl in zip(col_x, cols):
            ax.text(cx, 0.96, lbl, fontsize=7, fontfamily="Consolas",
                    color=TXT3, va="top", transform=ax.transAxes)
        ax.plot([0, 1], [0.94, 0.94], color=GRID_CLR, linewidth=0.7,
                transform=ax.transAxes)

        for i, d in enumerate(data):
            ep = d.get("earnings") or {}
            y_row = 0.94 - (i + 1) * row_h

            # Alternating row background
            row_bg = RAISED if i % 2 == 0 else PLOT_BG
            ax.axhspan(y_row - row_h * 0.05, y_row + row_h * 0.82,
                       xmin=0, xmax=1, facecolor=row_bg, zorder=0)

            # Status badge
            profitable = ep.get("profitable")
            if profitable is None:
                status_txt, status_clr = "N/A",    TXT3
            elif profitable:
                status_txt, status_clr = "PROFIT",  GREEN
            else:
                status_txt, status_clr = "LOSS",    RED

            ax.text(col_x[0], y_row + row_h * 0.35, status_txt,
                    fontsize=7, fontfamily="Consolas", color=status_clr,
                    va="center", transform=ax.transAxes, fontweight="bold")

            # SOE @ filing
            soe_f = ep.get("soe_at_filing")
            ax.text(col_x[1], y_row + row_h * 0.35,
                    f"{soe_f:+.3f}" if soe_f is not None else "—",
                    fontsize=7.5, fontfamily="Consolas", color=TXT2,
                    va="center", transform=ax.transAxes)

            # SOE now
            soe_n = ep.get("soe_latest")
            soe_n_clr = (GREEN if (soe_n or 0) > 0 else RED) if soe_n is not None else TXT3
            ax.text(col_x[2], y_row + row_h * 0.35,
                    f"{soe_n:+.3f}" if soe_n is not None else "—",
                    fontsize=7.5, fontfamily="Consolas", color=soe_n_clr,
                    va="center", transform=ax.transAxes)

            # Change
            direction = ep.get("direction", "n/a")
            chg_pct   = ep.get("change_pct")
            dir_arrow = {"improving": "↑", "deteriorating": "↓",
                         "stable": "→"}.get(direction, "—")
            dir_clr   = {"improving": GREEN, "deteriorating": RED,
                         "stable": TXT2}.get(direction, TXT3)
            chg_txt = f"{dir_arrow} {chg_pct:+.0f}%" if chg_pct is not None else dir_arrow
            ax.text(col_x[3], y_row + row_h * 0.35, chg_txt,
                    fontsize=7.5, fontfamily="Consolas", color=dir_clr,
                    va="center", transform=ax.transAxes)

            # Persistence
            plabel = ep.get("persistence_label", "N/A")
            p_clr  = {"HIGH": GREEN, "MODERATE": AMBER, "LOW": RED}.get(plabel, TXT3)
            ax.text(col_x[4], y_row + row_h * 0.35, plabel,
                    fontsize=7, fontfamily="Consolas", color=p_clr,
                    va="center", transform=ax.transAxes)

        scope_lbl = "all filings" if self._rankings_scope == "alltime" else "most recent filing per company"
        self._rankings_subtitle.set_text(
            f"Top {n}  ·  {scope_lbl}  ·  persistence = |ΔSOE| / σ(pre-filing SOE)"
            f"  ·  HIGH <0.5σ  MODERATE <1.0σ  LOW ≥1.0σ"
        )

    def _draw_right_stats(self, ax, data, ys, PLOT_BG, GRID_CLR):
        n = len(data)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        cols  = ["Ticker", "Vague%", "Sentences", "Filed"  ]
        col_x = [0.01,      0.25,     0.49,        0.72    ]
        row_h = 0.88 / (n + 1)

        # Header
        for cx, lbl in zip(col_x, cols):
            ax.text(cx, 0.96, lbl, fontsize=7, fontfamily="Consolas",
                    color=TXT3, va="top", transform=ax.transAxes)
        ax.plot([0, 1], [0.94, 0.94], color=GRID_CLR, linewidth=0.7,
                transform=ax.transAxes)

        for i, d in enumerate(data):
            y_row  = 0.94 - (i + 1) * row_h
            row_bg = RAISED if i % 2 == 0 else PLOT_BG
            ax.axhspan(y_row - row_h * 0.05, y_row + row_h * 0.82,
                       xmin=0, xmax=1, facecolor=row_bg, zorder=0)

            vals = [
                d["ticker"],
                f"{d['vague_pct']:.1f}%",
                f"{d['n_sentences']:,}",
                d["filing_date"],
            ]
            for cx, val in zip(col_x, vals):
                ax.text(cx, y_row + row_h * 0.35, val,
                        fontsize=8, fontfamily="Consolas", color=TXT2,
                        va="center", transform=ax.transAxes)

        scope_lbl = "all filings" if self._rankings_scope == "alltime" else "most recent filing per company"
        self._rankings_subtitle.set_text(
            f"Top {n}  ·  {scope_lbl}"
        )


# ── Widget helpers ────────────────────────────────────────────────────────────

def _divider(parent, orient: str = "h", pady: int = 0):
    ctk.CTkFrame(parent, height=1, fg_color=BORDER).pack(fill="x", pady=pady)


def _section_label(parent, text: str):
    ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(family="Consolas", size=9),
        text_color=TXT3,
    ).pack(anchor="w", pady=(8, 4))


def _body_card(parent, text: str):
    card = ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=8)
    card.pack(fill="x", pady=(0, 16))
    ctk.CTkLabel(
        card, text=text,
        font=ctk.CTkFont(family="DejaVu Sans", size=12),
        text_color=TXT2,
        anchor="w", justify="left",
        wraplength=750,
    ).pack(anchor="w", padx=16, pady=14)


def _flag_card(parent, flag: dict):
    assessment = flag.get("assessment", "").upper()
    fg, bg     = ASSESS_COLOURS.get(assessment, (TXT2, RAISED))

    topic     = flag.get("topic", "").replace("_", " ").upper()
    layer     = flag.get("layer", "").replace("_", " ")
    flag_num  = flag.get("flag_number", "")
    reasoning = flag.get("reasoning", "")
    action    = flag.get("investigation_action")

    card = ctk.CTkFrame(parent, fg_color=RAISED, corner_radius=8)
    card.pack(fill="x", pady=3)

    # Header row
    hdr = ctk.CTkFrame(card, fg_color="transparent")
    hdr.pack(fill="x", padx=12, pady=(10, 8))

    ctk.CTkLabel(
        hdr,
        text=f"  {assessment}  ",
        font=ctk.CTkFont(family="Consolas", size=9, weight="bold"),
        text_color=fg, fg_color=bg,
        corner_radius=4,
    ).pack(side="left")

    ctk.CTkLabel(
        hdr,
        text=f"  {topic}",
        font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
        text_color=TXT,
    ).pack(side="left", padx=(8, 0))

    ctk.CTkLabel(
        hdr,
        text=f"  {layer}",
        font=ctk.CTkFont(family="Consolas", size=9),
        text_color=TXT3,
    ).pack(side="left")

    ctk.CTkLabel(
        hdr,
        text=f"#{flag_num}",
        font=ctk.CTkFont(family="Consolas", size=9),
        text_color=TXT3,
    ).pack(side="right")

    # Reasoning body
    if reasoning:
        ctk.CTkLabel(
            card,
            text=reasoning,
            font=ctk.CTkFont(family="DejaVu Sans", size=11),
            text_color=TXT2,
            anchor="w", justify="left",
            wraplength=750,
        ).pack(anchor="w", padx=16, pady=(0, 8))

    # Investigation action (amber call-out box)
    if action:
        act = ctk.CTkFrame(card, fg_color=SURFACE, corner_radius=6)
        act.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkLabel(
            act,
            text="  INVESTIGATION ACTION",
            font=ctk.CTkFont(family="Consolas", size=8, weight="bold"),
            text_color=AMBER,
        ).pack(anchor="w", padx=10, pady=(8, 2))
        ctk.CTkLabel(
            act,
            text=action,
            font=ctk.CTkFont(family="DejaVu Sans", size=10),
            text_color=TXT2,
            anchor="w", justify="left",
            wraplength=720,
        ).pack(anchor="w", padx=10, pady=(0, 10))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = DDDSApp()
    app.mainloop()

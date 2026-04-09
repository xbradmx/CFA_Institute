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

import json
import os
import sys
import threading
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

        self._build_layout()
        self._populate_ticker_list(self._tickers)
        # Start loading heatmap data in background immediately
        threading.Thread(target=self._load_heatmap, daemon=True).start()

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

        self._build_memo_tab()
        self._build_heatmap_tab()

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
            from run_10_risk_heatmap import build_figure
            fig = build_figure(str(_OUTPUTS_DIR), dpi=110)
            self.after(0, lambda: self._embed_heatmap(fig))
        except Exception as exc:
            self.after(0, lambda: self._heatmap_spinner.configure(
                text=f"Heatmap unavailable.\n{exc}",
                wraplength=500,
            ))

    def _embed_heatmap(self, fig):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self._heatmap_spinner.place_forget()
        canvas = FigureCanvasTkAgg(fig, master=self._heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


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

"""
DDDS — Disclosure Degradation Detection System
Region Heat Risk Map  (Desktop Application)

Bloomberg-inspired dark-mode global heat map. When pipeline data is
available (data/findings/*.json or data/passages.csv), the map renders
actual DDDS risk scores derived from Run 8 findings. Falls back to
demo mode with synthetic scores if no pipeline data is present.

Score derivation from pipeline data
------------------------------------
Each company in data/findings/ contributes a risk score:
    HIGH signal    → 0.85
    MEDIUM signal  → 0.55
    LOW signal     → 0.25
    No findings    → 0.10

Scores are aggregated by SIC sector group and mapped to geographic
regions using the ISO alpha-3 country code on the world map. Since all
266 companies are US Industrials (SIC 3400-3599), the US is always the
primary focal region. The map also shows geographic risk signals
extracted from the text of Risk Factor passages (references to specific
countries/regions).

Requirements
------------
    pip install geopandas geodatasets matplotlib shapely python-dotenv

Run
---
    python ddds_heat_map.py
    python ddds_heat_map.py --findings-dir data/findings
    python ddds_heat_map.py --demo          # random scores, no pipeline data needed
"""

import argparse
import glob
import json
import os
import random
import sys
import tkinter as tk

import geopandas as gpd
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FINDINGS_DIR = os.environ.get("FINDINGS_DIR", "data/findings")
PASSAGES_CSV = os.environ.get("PASSAGES_CSV", "data/passages.csv")

# Signal → numeric score mapping
SIGNAL_SCORES = {
    "HIGH":   0.85,
    "MEDIUM": 0.55,
    "LOW":    0.25,
}

# Geographic risk keywords — maps country mentions in risk text to ISO codes
# Extend this dict as needed for your coverage universe
GEO_KEYWORDS = {
    "china":         "CHN",
    "chinese":       "CHN",
    "mexico":        "MEX",
    "mexican":       "MEX",
    "canada":        "CAN",
    "canadian":      "CAN",
    "germany":       "DEU",
    "german":        "DEU",
    "japan":         "JPN",
    "japanese":      "JPN",
    "india":         "IND",
    "indian":        "IND",
    "brazil":        "BRA",
    "brazil":        "BRA",
    "taiwan":        "TWN",
    "south korea":   "KOR",
    "korea":         "KOR",
    "vietnam":       "VNM",
    "france":        "FRA",
    "france":        "FRA",
    "uk":            "GBR",
    "united kingdom":"GBR",
    "russia":        "RUS",
    "russian":       "RUS",
    "ukraine":       "UKR",
    "middle east":   None,   # region — skip
    "europe":        None,   # region — skip
    "asia":          None,   # region — skip
}
# ---------------------------------------------------------------------------


# ── Palette ─────────────────────────────────────────────────────────────────
GREY_ZERO    = "#2d3340"
BG_DARK      = "#0d1117"
BG_PANEL     = "#161b22"
BORDER_CLR   = "#30363d"
BORDER_MAP   = "#ffffff"
TEXT_PRIMARY = "#e0e4ec"
TEXT_DIM     = "#484f5a"
ACCENT       = "#58a6ff"

_stops = [
    (0.00, "#16a34a"),
    (0.22, "#65a30d"),
    (0.44, "#ca8a04"),
    (0.67, "#ea580c"),
    (1.00, "#dc2626"),
]


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


RISK_CMAP = LinearSegmentedColormap.from_list(
    "ddds", [(p, _hex(c)) for p, c in _stops]
)


def score_colour(s):
    if s == 0:
        return GREY_ZERO
    t = max(0.0, min(1.0, (s - 0.1) / 0.9))
    r, g, b, _ = RISK_CMAP(t)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ── Map data ────────────────────────────────────────────────────────────────

def _load_world() -> gpd.GeoDataFrame:
    """
    Load world geometries. Tries geodatasets first (geopandas >= 1.0),
    falls back to the bundled naturalearth_lowres for older versions.
    """
    # Method 1: geodatasets (recommended for geopandas >= 1.0)
    try:
        import geodatasets
        path = geodatasets.get_path("naturalearth.land")
        # geodatasets 'land' has no country names — use naturalearth countries
        path = geodatasets.get_path("naturalearth.countries")
        world = gpd.read_file(path)
        # Standardise column names
        if "NAME" in world.columns and "name" not in world.columns:
            world = world.rename(columns={"NAME": "name"})
        if "ISO_A3" in world.columns and "iso_a3" not in world.columns:
            world = world.rename(columns={"ISO_A3": "iso_a3"})
        return world
    except Exception:
        pass

    # Method 2: geopandas bundled dataset (geopandas < 1.0)
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        if "iso_a3" not in world.columns:
            world["iso_a3"] = ""
        return world
    except Exception:
        pass

    # Method 3: download directly from Natural Earth via URL
    try:
        url = (
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        )
        world = gpd.read_file(url)
        world = world.rename(columns={"NAME": "name", "ISO_A3": "iso_a3"})
        return world
    except Exception as e:
        raise RuntimeError(
            f"Could not load world map data. Install geodatasets:\n"
            f"  pip install geodatasets\n"
            f"Original error: {e}"
        )


world = _load_world()
world = world[world.get("name", world.get("NAME", "")) != "Antarctica"].copy()
world.reset_index(drop=True, inplace=True)

# Ensure iso_a3 column exists
if "iso_a3" not in world.columns:
    world["iso_a3"] = ""

# ── Score loading ───────────────────────────────────────────────────────────

def load_pipeline_scores(findings_dir: str) -> tuple[dict, dict, int]:
    """
    Reads all *_findings.json files from findings_dir.

    Returns
    -------
    iso_scores   : dict[iso_a3 -> float]  country-level aggregated scores
    country_detail: dict[iso_a3 -> list[str]]  company names per country
    company_count : int  total companies with findings
    """
    pattern = os.path.join(findings_dir, "*_findings.json")
    files   = glob.glob(pattern)

    if not files:
        return {}, {}, 0

    # US always gets the aggregate of all findings (all 266 are US companies)
    us_scores = []

    # Geographic signals from risk text — secondary scores for other countries
    geo_scores = {}

    company_count = 0

    for path in files:
        try:
            with open(path) as f:
                findings = json.load(f)
        except Exception:
            continue

        company_count += 1
        signal = (
            findings.get("deep_analysis", {})
                    .get("overall_signal", {})
                    .get("signal_strength", "LOW")
        )
        score = SIGNAL_SCORES.get(signal, 0.10)
        us_scores.append(score)

        # Scan flagged items text for geographic mentions
        for item in findings.get("flagged_items", []):
            text = (item.get("curr_text", "") + " " +
                    item.get("prev_text", "")).lower()
            for keyword, iso in GEO_KEYWORDS.items():
                if iso and keyword in text:
                    geo_scores.setdefault(iso, []).append(score * 0.6)

    iso_scores    = {}
    country_detail = {}

    # US gets the mean of all company scores
    if us_scores:
        iso_scores["USA"] = round(sum(us_scores) / len(us_scores), 2)
        country_detail["USA"] = [
            f"{len(us_scores)} US Industrials companies analysed",
            f"Mean signal score: {iso_scores['USA']:.2f}",
            f"High signals: {sum(1 for s in us_scores if s >= 0.85)}",
            f"Medium signals: {sum(1 for s in us_scores if 0.45 <= s < 0.85)}",
        ]

    # Other countries get reduced geographic risk scores
    for iso, scores in geo_scores.items():
        iso_scores[iso]    = round(min(sum(scores) / len(scores), 0.75), 2)
        country_detail[iso] = [f"Referenced in {len(scores)} filing(s) as geographic risk"]

    return iso_scores, country_detail, company_count


def apply_scores(iso_scores: dict, demo_mode: bool):
    """Applies scores to the world GeoDataFrame."""
    if demo_mode or not iso_scores:
        world["score"]  = [round(random.randint(0, 10) / 10, 1) for _ in range(len(world))]
    else:
        def lookup(row):
            iso = row.get("iso_a3", "")
            return iso_scores.get(iso, 0.0)
        world["score"] = world.apply(lookup, axis=1)

    world["colour"] = world["score"].apply(score_colour)


# Spatial index for hover
from shapely.strtree import STRtree

_tree  = STRtree(world.geometry.values)
_geoms = list(world.geometry.values)


def country_at(x, y):
    from shapely.geometry import Point
    pt   = Point(x, y)
    hits = _tree.query(pt)
    for idx in hits:
        if _geoms[idx].contains(pt):
            row = world.iloc[idx]
            return row.get("name", ""), row.get("iso_a3", ""), row["score"]
    return None


# ── Application ──────────────────────────────────────────────────────────────

class DDDSApp:
    def __init__(self, root, findings_dir: str, demo_mode: bool):
        self.root         = root
        self.findings_dir = findings_dir
        self.demo_mode    = demo_mode
        self.iso_scores   = {}
        self.country_detail = {}
        self.company_count  = 0

        self.root.title("Disclosure Degradation Detection System")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1360x780")
        self.root.minsize(900, 550)

        self._load_data()
        self._build_header()
        self._build_map()
        self._build_footer()
        self.draw_map()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._last_hover = None

    def _load_data(self):
        if not self.demo_mode:
            self.iso_scores, self.country_detail, self.company_count = \
                load_pipeline_scores(self.findings_dir)
            if not self.iso_scores:
                print(
                    f"  [!] No findings JSON files found in '{self.findings_dir}'.\n"
                    f"      Running in demo mode with synthetic scores.\n"
                    f"      Run graph_rag.py first to generate real data."
                )
                self.demo_mode = True
        apply_scores(self.iso_scores, self.demo_mode)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=58)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        left = tk.Frame(header, bg=BG_PANEL)
        left.pack(side="left", padx=24, pady=8)

        tk.Label(left, text="DISCLOSURE DEGRADATION DETECTION SYSTEM",
                 font=("Consolas", 8), fg=TEXT_DIM, bg=BG_PANEL
                 ).pack(anchor="w")
        tk.Label(left, text="Region Heat Risk Map",
                 font=("Helvetica", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_PANEL
                 ).pack(anchor="w")

        right = tk.Frame(header, bg=BG_PANEL)
        right.pack(side="right", padx=24, pady=8)

        mode_text = "DEMO MODE" if self.demo_mode else f"{self.company_count} companies"
        mode_clr  = ACCENT if not self.demo_mode else "#ea580c"

        self.count_lbl = tk.Label(
            right, text=mode_text,
            font=("Consolas", 9), fg=mode_clr, bg=BG_PANEL)
        self.count_lbl.pack(side="left", padx=(0, 14))

        tk.Button(
            right, text="↻  Refresh", font=("Helvetica", 9),
            fg=ACCENT, bg=BG_PANEL, activeforeground="#79c0ff",
            activebackground="#1c2128", bd=0, padx=12, pady=4,
            relief="flat", cursor="hand2", command=self._refresh,
        ).pack(side="left")

        tk.Frame(self.root, bg=BORDER_CLR, height=1).pack(fill="x")

    def _build_map(self):
        frame = tk.Frame(self.root, bg=BG_DARK)
        frame.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(
            figsize=(14, 7), dpi=120, facecolor=BG_DARK,
        )
        self.ax.set_facecolor(BG_DARK)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(self.root, bg=BG_PANEL)
        tb_frame.pack(fill="x", side="bottom")
        tb = NavigationToolbar2Tk(self.canvas, tb_frame)
        tb.config(bg=BG_PANEL)
        tb.update()
        for w in tb.winfo_children():
            try:
                w.config(bg=BG_PANEL, highlightbackground=BG_PANEL)
            except tk.TclError:
                pass

    def _build_footer(self):
        ft = tk.Frame(self.root, bg=BG_PANEL, height=26)
        ft.pack(fill="x", side="bottom")
        ft.pack_propagate(False)
        data_label = (
            "DEMO — synthetic scores"
            if self.demo_mode
            else f"Pipeline data from: {self.findings_dir}"
        )
        tk.Label(
            ft,
            text=f"DDDS v2.1  ·  CFA AI Investment Challenge 2026  ·  {data_label}",
            font=("Consolas", 8), fg=TEXT_DIM, bg=BG_PANEL
        ).pack(side="right", padx=16)

    # ── Drawing ──────────────────────────────────────────────────────────

    def draw_map(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal")

        world.plot(ax=ax, color=world["colour"], linewidth=0, antialiased=True)
        world.boundary.plot(
            ax=ax, edgecolor=BORDER_MAP, linewidth=0.55, alpha=0.30, antialiased=True,
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        # Colour bar
        sm = plt.cm.ScalarMappable(cmap=RISK_CMAP, norm=plt.Normalize(0.1, 1.0))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.02, aspect=30)
        cbar.set_label(
            "DISCLOSURE RISK SCORE", fontsize=8, color=TEXT_DIM,
            fontfamily="Consolas", labelpad=8
        )
        cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 1.0])
        cbar.set_ticklabels(["LOW", "0.3", "0.5", "0.7", "HIGH"])
        cbar.ax.tick_params(labelsize=8, colors=TEXT_DIM, length=0)
        cbar.outline.set_edgecolor(BORDER_CLR)
        cbar.outline.set_linewidth(0.5)

        # Unscored legend patch
        ax.add_patch(plt.Rectangle((-178, -57), 3, 4, fc=GREY_ZERO, ec="none", zorder=5))
        ax.text(-174, -54.5, "0.0 — Unscored", fontsize=8,
                fontfamily="Consolas", color=TEXT_DIM, ha="left", va="center")

        # Data mode watermark
        if self.demo_mode:
            ax.text(0, 0, "DEMO MODE — SYNTHETIC SCORES",
                    ha="center", va="center", fontsize=14,
                    fontfamily="Consolas", color="#ea580c", alpha=0.18,
                    rotation=0, transform=ax.transAxes,
                    fontweight="bold")

        # Tooltip
        self.annot = ax.annotate(
            "", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
            fontsize=10, fontfamily="Helvetica", fontweight="bold",
            color=TEXT_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.55", fc=BG_PANEL,
                      ec=BORDER_CLR, lw=0.8, alpha=0.96),
            arrowprops=dict(arrowstyle="->", color=BORDER_CLR, lw=0.8),
            zorder=100, visible=False,
        )

        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    # ── Interactivity ─────────────────────────────────────────────────────

    def _refresh(self):
        """Reloads pipeline data and redraws — or regenerates demo scores."""
        self._load_data()
        self.draw_map()
        self._last_hover = None

    def _on_hover(self, event):
        if event.inaxes != self.ax:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()
            self._last_hover = None
            return

        result = country_at(event.xdata, event.ydata)

        if result:
            name, iso, score = result

            if self._last_hover == name:
                self.annot.xy = (event.xdata, event.ydata)
                self.canvas.draw_idle()
                return

            self._last_hover = name
            clr = score_colour(score) if score > 0 else TEXT_DIM

            # Build tooltip text
            lines = [f"{name}   Score: {score:.2f}"]
            if iso and iso in self.country_detail:
                for detail in self.country_detail[iso][:3]:
                    lines.append(f"  {detail}")

            self.annot.xy = (event.xdata, event.ydata)
            self.annot.set_text("\n".join(lines))
            self.annot.set_color(clr)
            self.annot.set_visible(True)
        else:
            if self.annot.get_visible():
                self.annot.set_visible(False)
            self._last_hover = None

        self.canvas.draw_idle()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DDDS Region Heat Risk Map")
    parser.add_argument(
        "--findings-dir", default=FINDINGS_DIR,
        help="Directory containing *_findings.json files from Run 8"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic random scores (no pipeline data required)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    root = tk.Tk()

    # Dark title bar on Windows 10/11
    try:
        import ctypes
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
        )
    except Exception:
        pass

    app = DDDSApp(root, findings_dir=args.findings_dir, demo_mode=args.demo)
    root.mainloop()
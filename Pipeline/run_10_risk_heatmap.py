"""
DDDS — Disclosure Degradation Detection System
Region Heat Risk Map  (Desktop Application)

Bloomberg-inspired dark-mode global heat map driven by raw vagueness scores
extracted from the analyst_outputs CSV files produced by Run 9.

Score derivation
----------------
For each sentence in every *_ddds_scores.csv file:
  - If the sentence text mentions a recognised country/region keyword, the
    sentence's raw vague_prob is attributed to that country.
  - The USA always receives scores from ALL sentences across all filings
    (since the entire coverage universe is US Industrials).

Per-country score = mean(raw vague_prob) across all attributed sentences.

No demeaning, no bucketing.  A country where sentences tend to be more vague
(higher vague_prob) will present as a greater heat on the map — exactly as the
classifier produces it.

Requirements
------------
    pip install geopandas geodatasets matplotlib shapely python-dotenv

Run
---
    python run_10_risk_heat_map.py
    python run_10_risk_heat_map.py --analyst-outputs-dir data/analyst_outputs
    python run_10_risk_heat_map.py --demo

    # Export a PNG for presentations (also opens the interactive window):
    python run_10_risk_heat_map.py --save outputs/heatmap.png

    # Headless export only — no Tkinter window (ideal for CI / pre-demo prep):
    python run_10_risk_heat_map.py --save outputs/heatmap.png --headless

    # Demo mode headless export:
    python run_10_risk_heat_map.py --demo --save outputs/heatmap_demo.png --headless
"""

import argparse
import csv
import glob
import io
import os
import random
import sys
import tkinter as tk
import geopandas as gpd
import matplotlib
matplotlib.rcParams["agg.path.chunksize"] = 10000
# Backend is set later in parse_args — must happen before pyplot import
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ANALYST_OUTPUTS_DIR = os.environ.get("ANALYST_OUTPUTS_DIR", "data/analyst_outputs")

# Geographic keywords → ISO alpha-3.  None = broad region, skip.
# Extend as needed for your coverage universe.
GEO_KEYWORDS: dict[str, str | None] = {
    # North America
    "canada":           "CAN",
    "canadian":         "CAN",
    "mexico":           "MEX",
    "mexican":          "MEX",
    # Europe
    "germany":          "DEU",
    "german":           "DEU",
    "france":           "FRA",
    "french":           "FRA",
    "united kingdom":   "GBR",
    "uk":               "GBR",
    "britain":          "GBR",
    "british":          "GBR",
    "italy":            "ITA",
    "italian":          "ITA",
    "spain":            "ESP",
    "spanish":          "ESP",
    "netherlands":      "NLD",
    "dutch":            "NLD",
    "sweden":           "SWE",
    "swedish":          "SWE",
    "poland":           "POL",
    "polish":           "POL",
    "ukraine":          "UKR",
    "ukrainian":        "UKR",
    "russia":           "RUS",
    "russian":          "RUS",
    # Asia-Pacific
    "china":            "CHN",
    "chinese":          "CHN",
    "japan":            "JPN",
    "japanese":         "JPN",
    "south korea":      "KOR",
    "korea":            "KOR",
    "korean":           "KOR",
    "taiwan":           "TWN",
    "india":            "IND",
    "indian":           "IND",
    "vietnam":          "VNM",
    "vietnamese":       "VNM",
    "malaysia":         "MYS",
    "singapore":        "SGP",
    "indonesia":        "IDN",
    "thailand":         "THA",
    "australia":        "AUS",
    "australian":       "AUS",
    # Middle East / Africa
    "saudi arabia":     "SAU",
    "saudi":            "SAU",
    "israel":           "ISR",
    "israeli":          "ISR",
    "uae":              "ARE",
    "south africa":     "ZAF",
    # Latin America
    "brazil":           "BRA",
    "brazilian":        "BRA",
    "argentina":        "ARG",
    "chilean":          "CHL",
    "chile":            "CHL",
    "colombia":         "COL",
    "colombian":        "COL",
    # Democratic Republic of Congo (conflict minerals)
    "congo":            "COD",
    # Broad regions — tracked in tooltip but not mapped to a single country
    "middle east":      None,
    "europe":           None,
    "asia":             None,
    "latin america":    None,
    "africa":           None,
}

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
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


def _hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


RISK_CMAP = LinearSegmentedColormap.from_list(
    "ddds", [(p, _hex(c)) for p, c in _stops]
)


def score_colour(s: float) -> str:
    if s == 0:
        return GREY_ZERO
    t = max(0.0, min(1.0, s))           # vague_prob is already 0–1
    r, g, b, _ = RISK_CMAP(t)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ---------------------------------------------------------------------------
# World map loading
# ---------------------------------------------------------------------------

def _load_world() -> gpd.GeoDataFrame:
    try:
        import geodatasets
        path = geodatasets.get_path("naturalearth.countries")
        world = gpd.read_file(path)
        if "NAME" in world.columns and "name" not in world.columns:
            world = world.rename(columns={"NAME": "name"})
        if "ISO_A3" in world.columns and "iso_a3" not in world.columns:
            world = world.rename(columns={"ISO_A3": "iso_a3"})
        return world
    except Exception:
        pass

    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        if "iso_a3" not in world.columns:
            world["iso_a3"] = ""
        return world
    except Exception:
        pass

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
if "iso_a3" not in world.columns:
    world["iso_a3"] = ""


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _parse_analyst_csv(path: str) -> list[dict]:
    """
    Parse a *_ddds_scores.csv file.

    The format is:
        Lines 1-N : document summary block (free-form)
        SENTENCE LEVEL DETAIL
        sentence_id,section,text,vague_prob,...
        <data rows>

    Returns a list of dicts with keys: text, vague_prob, complex_prob, section.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Locate the sentence-level block
    marker = "SENTENCE LEVEL DETAIL"
    idx = raw.find(marker)
    if idx == -1:
        return []

    # Everything after the marker
    remainder = raw[idx + len(marker):].lstrip("\r\n")

    rows = []
    reader = csv.DictReader(io.StringIO(remainder))
    for row in reader:
        try:
            vague_prob   = float(row.get("vague_prob",   0))
            complex_prob = float(row.get("complex_prob", 0))
        except (ValueError, TypeError):
            continue
        rows.append({
            "text":         row.get("text", "").lower(),
            "section":      row.get("section", ""),
            "vague_prob":   vague_prob,
            "complex_prob": complex_prob,
        })
    return rows


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

def load_scores_from_csvs(outputs_dir: str) -> tuple[dict, dict, int]:
    """
    Reads all *_ddds_scores.csv files from outputs_dir.

    Returns
    -------
    iso_scores    : dict[iso_a3 -> float]   mean raw vague_prob per country
    country_detail: dict[iso_a3 -> list[str]] tooltip lines per country
    file_count    : int  number of CSV files processed
    """
    pattern = os.path.join(outputs_dir, "*_ddds_scores.csv")
    files   = glob.glob(pattern)

    if not files:
        return {}, {}, 0

    # Accumulate vague_prob lists keyed by ISO code
    # USA gets every sentence; other countries get sentences that mention them
    geo_scores: dict[str, list[float]] = {}
    us_scores:  list[float] = []

    # For tooltip detail: count mentions per country
    mention_counts: dict[str, int] = {}

    file_count = 0

    for path in files:
        rows = _parse_analyst_csv(path)
        if not rows:
            continue

        file_count += 1

        for row in rows:
            vp   = row["vague_prob"]
            text = row["text"]

            # US gets every sentence
            us_scores.append(vp)

            # Check for geographic mentions
            for keyword, iso in GEO_KEYWORDS.items():
                if iso and keyword in text:
                    geo_scores.setdefault(iso, []).append(vp)
                    mention_counts[iso] = mention_counts.get(iso, 0) + 1

    iso_scores    = {}
    country_detail = {}

    if us_scores:
        us_mean = sum(us_scores) / len(us_scores)
        iso_scores["USA"] = round(us_mean, 4)
        vague_count = sum(1 for s in us_scores if s >= 0.5)
        country_detail["USA"] = [
            f"{file_count} US Industrials filings analysed",
            f"Mean vague_prob (all sentences): {us_mean:.3f}",
            f"Sentences ≥ 0.5 vague_prob: {vague_count} / {len(us_scores)}",
        ]

    for iso, scores in geo_scores.items():
        mean_score = sum(scores) / len(scores)
        iso_scores[iso]     = round(mean_score, 4)
        country_detail[iso] = [
            f"Mentioned in {mention_counts.get(iso, len(scores))} sentence(s)",
            f"Mean vague_prob of those sentences: {mean_score:.3f}",
            f"Sample size: {len(scores)} sentence(s)",
        ]

    return iso_scores, country_detail, file_count


# ---------------------------------------------------------------------------
# Apply scores to GeoDataFrame
# ---------------------------------------------------------------------------

def apply_scores(iso_scores: dict, demo_mode: bool):
    if demo_mode or not iso_scores:
        world["score"]  = [round(random.random(), 3) for _ in range(len(world))]
    else:
        def lookup(row):
            return iso_scores.get(row.get("iso_a3", ""), 0.0)
        world["score"] = world.apply(lookup, axis=1)

    world["colour"] = world["score"].apply(score_colour)


# ---------------------------------------------------------------------------
# Spatial hover index
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DDDSApp:
    def __init__(self, root: tk.Tk, outputs_dir: str, demo_mode: bool):
        self.root        = root
        self.outputs_dir = outputs_dir
        self.demo_mode   = demo_mode

        self.iso_scores     = {}
        self.country_detail = {}
        self.file_count     = 0

        self.root.title("Disclosure Degradation Detection System")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1360x780")
        self.root.minsize(900, 550)

        self._load_data()
        self._build_header()
        self._build_map()
        self._build_footer()
        self.draw_map()

    # ── Data ──────────────────────────────────────────────────────────────

    def _load_data(self):
        if not self.demo_mode:
            self.iso_scores, self.country_detail, self.file_count = \
                load_scores_from_csvs(self.outputs_dir)
            if not self.iso_scores:
                print(
                    f"  [!] No *_ddds_scores.csv files found in '{self.outputs_dir}'.\n"
                    f"      Falling back to demo mode with synthetic scores.\n"
                    f"      Run the analyst output pipeline first to generate real data."
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

        tk.Label(
            left, text="DISCLOSURE DEGRADATION DETECTION SYSTEM",
            font=("Consolas", 8), fg=TEXT_DIM, bg=BG_PANEL,
        ).pack(anchor="w")
        tk.Label(
            left, text="Region Vagueness Heat Map",
            font=("DejaVu Sans", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_PANEL,
        ).pack(anchor="w")

        right = tk.Frame(header, bg=BG_PANEL)
        right.pack(side="right", padx=24, pady=8)

        mode_text = "DEMO MODE" if self.demo_mode else f"{self.file_count} filing(s)"
        mode_clr  = ACCENT if not self.demo_mode else "#ea580c"

        self.count_lbl = tk.Label(
            right, text=mode_text,
            font=("Consolas", 9), fg=mode_clr, bg=BG_PANEL,
        )
        self.count_lbl.pack(side="left", padx=(0, 14))

        tk.Button(
            right, text="↻  Refresh",
            font=("DejaVu Sans", 9),
            fg=ACCENT, bg=BG_PANEL,
            activeforeground="#79c0ff", activebackground="#1c2128",
            bd=0, padx=12, pady=4,
            relief="flat", cursor="hand2",
            command=self._refresh,
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

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._last_hover = None

    def _build_footer(self):
        ft = tk.Frame(self.root, bg=BG_PANEL, height=26)
        ft.pack(fill="x", side="bottom")
        ft.pack_propagate(False)
        data_label = (
            "DEMO — synthetic scores"
            if self.demo_mode
            else f"Raw vague_prob scores from: {self.outputs_dir}"
        )
        tk.Label(
            ft,
            text=f"DDDS v2.2  ·  CFA AI Investment Challenge 2026  ·  {data_label}",
            font=("Consolas", 8), fg=TEXT_DIM, bg=BG_PANEL,
        ).pack(side="right", padx=16)

    # ── Drawing ───────────────────────────────────────────────────────────

    def draw_map(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal")

        world.plot(ax=ax, color=world["colour"], linewidth=0, antialiased=True)
        world.boundary.plot(
            ax=ax, edgecolor=BORDER_MAP, linewidth=0.55, alpha=0.30,
            antialiased=True,
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        # Colour bar — scale is 0.0 to 1.0 (raw vague_prob range)
        sm = plt.cm.ScalarMappable(
            cmap=RISK_CMAP, norm=plt.Normalize(0.0, 1.0)
        )
        sm.set_array([])
        cbar = self.fig.colorbar(
            sm, ax=ax, fraction=0.012, pad=0.02, aspect=30
        )
        cbar.set_label(
            "MEAN RAW VAGUE_PROB", fontsize=8, color=TEXT_DIM,
            fontfamily="Consolas", labelpad=8,
        )
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
        cbar.ax.tick_params(labelsize=8, colors=TEXT_DIM, length=0)
        cbar.outline.set_edgecolor(BORDER_CLR)
        cbar.outline.set_linewidth(0.5)

        # Unscored legend patch
        ax.add_patch(
            plt.Rectangle((-178, -57), 3, 4, fc=GREY_ZERO, ec="none", zorder=5)
        )
        ax.text(
            -174, -54.5, "0.0 — Unscored",
            fontsize=8, fontfamily="Consolas",
            color=TEXT_DIM, ha="left", va="center",
        )

        # Score source annotation (top-left)
        ax.text(
            0.01, 0.99,
            "Score = mean raw vague_prob of sentences mentioning each geography",
            transform=ax.transAxes,
            fontsize=7, fontfamily="Consolas",
            color=TEXT_DIM, ha="left", va="top",
        )

        if self.demo_mode:
            ax.text(
                0.5, 0.5, "DEMO MODE — SYNTHETIC SCORES",
                ha="center", va="center", fontsize=14,
                fontfamily="Consolas", color="#ea580c", alpha=0.18,
                rotation=0, transform=ax.transAxes, fontweight="bold",
            )

        # Tooltip annotation
        self.annot = ax.annotate(
            "", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
            fontsize=10, fontfamily="DejaVu Sans", fontweight="bold",
            color=TEXT_PRIMARY,
            bbox=dict(
                boxstyle="round,pad=0.55", fc=BG_PANEL,
                ec=BORDER_CLR, lw=0.8, alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="->", color=BORDER_CLR, lw=0.8),
            zorder=100, visible=False,
        )

        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    # ── Interactivity ─────────────────────────────────────────────────────

    def _refresh(self):
        self._load_data()
        mode_text = "DEMO MODE" if self.demo_mode else f"{self.file_count} filing(s)"
        self.count_lbl.config(
            text=mode_text,
            fg="#ea580c" if self.demo_mode else ACCENT,
        )
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

            lines = [f"{name}   vague_prob: {score:.3f}"]
            if iso and iso in self.country_detail:
                for detail in self.country_detail[iso][:4]:
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



# ---------------------------------------------------------------------------
# Embeddable figure builder  (used by analyst_frontend.py)
# ---------------------------------------------------------------------------

def build_figure(analyst_outputs_dir: str = None, dpi: int = 120):
    """
    Build and return (fig, on_hover) for the geographic risk heatmap.

    Designed for embedding in a GUI via FigureCanvasTkAgg.  Does not call
    plt.show() or open any window.  The caller must connect on_hover:

        canvas.mpl_connect("motion_notify_event", on_hover)

    Parameters
    ----------
    analyst_outputs_dir : str, optional
        Path to the directory containing ``*_ddds_scores.csv`` files.
    dpi : int
        Figure DPI. Default 120.

    Returns
    -------
    (matplotlib.figure.Figure, callable)
    """
    from matplotlib.figure import Figure
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    outputs_dir = analyst_outputs_dir or ANALYST_OUTPUTS_DIR
    iso_scores, country_detail, file_count = load_scores_from_csvs(outputs_dir)
    apply_scores(iso_scores, demo_mode=not iso_scores)

    OCEAN_CLR   = "#0a1628"
    BORDER_LAND = "#1e2d40"

    fig = Figure(figsize=(14, 7), dpi=dpi, facecolor=BG_DARK)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(OCEAN_CLR)
    ax.set_aspect("equal")

    # Subtle graticule lines
    for lat in range(-60, 91, 30):
        ax.axhline(lat, color="#ffffff", alpha=0.03, linewidth=0.5, zorder=1)
    for lon in range(-180, 181, 60):
        ax.axvline(lon, color="#ffffff", alpha=0.03, linewidth=0.5, zorder=1)

    world.plot(ax=ax, color=world["colour"], linewidth=0, antialiased=True, zorder=2)
    world.boundary.plot(
        ax=ax, edgecolor=BORDER_LAND, linewidth=0.5, alpha=0.8,
        antialiased=True, zorder=3,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=RISK_CMAP, norm=Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.010, pad=0.01, aspect=40, shrink=0.65)
    cbar.set_label(
        "DISCLOSURE VAGUENESS", fontsize=7, color=TEXT_DIM,
        fontfamily="Consolas", labelpad=10,
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["LOW", "MED", "HIGH"])
    cbar.ax.tick_params(labelsize=7, colors=TEXT_DIM, length=0)
    cbar.outline.set_edgecolor(BORDER_CLR)
    cbar.outline.set_linewidth(0.5)

    # Unscored legend swatch
    ax.add_patch(Rectangle((-178, -57), 3, 4, fc=GREY_ZERO, ec=BORDER_CLR,
                            linewidth=0.5, zorder=5))
    ax.text(-174, -55, "No data", fontsize=7, fontfamily="Consolas",
            color=TEXT_DIM, ha="left", va="center", zorder=6)

    # Footer
    label = (
        f"{file_count} filing(s)  ·  mean vague_prob per geography"
        if file_count else "Demo mode  ·  synthetic scores"
    )
    ax.text(0.5, 0.01, label, transform=ax.transAxes, fontsize=7,
            fontfamily="Consolas", color=TEXT_DIM, ha="center", va="bottom")

    # Tooltip annotation (hidden until hover)
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        fontsize=9, fontfamily="DejaVu Sans",
        color=TEXT_PRIMARY,
        bbox=dict(boxstyle="round,pad=0.65", fc="#0e1520",
                  ec=BORDER_CLR, lw=1.2, alpha=0.97),
        arrowprops=dict(arrowstyle="-", color=BORDER_CLR, lw=0.6),
        zorder=100, visible=False,
    )

    fig.tight_layout(pad=0.4)

    # Closure — captures ax, annot, country_detail
    def on_hover(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                event.canvas.draw_idle()
            return

        result = country_at(event.xdata, event.ydata)
        if result:
            name, iso, score = result
            clr = score_colour(score) if score > 0 else "#3d4d63"
            risk = (
                "HIGH"  if score >= 0.67 else
                "MED"   if score >= 0.33 else
                "LOW"   if score >  0    else
                "—"
            )
            lines = [f"{name}   [{risk}]", f"Vague prob: {score:.3f}"]
            if iso and iso in country_detail:
                for d in country_detail[iso][:3]:
                    lines.append(f"  {d}")

            annot.xy = (event.xdata, event.ydata)
            annot.set_text("\n".join(lines))
            annot.get_bbox_patch().set_edgecolor(clr)
            annot.set_visible(True)
        else:
            if annot.get_visible():
                annot.set_visible(False)

        event.canvas.draw_idle()

    return fig, on_hover


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DDDS Region Vagueness Heat Map")
    parser.add_argument(
        "--analyst-outputs-dir",
        default=ANALYST_OUTPUTS_DIR,
        help="Directory containing *_ddds_scores.csv files from the analyst pipeline",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic random scores (no pipeline data required)",
    )
    parser.add_argument(
        "--save", metavar="PATH", default=None,
        help=(
            "Export the rendered map to a file. Supports .png, .svg, .pdf. "
            "Example: --save outputs/heatmap.png"
        ),
    )
    parser.add_argument(
        "--headless", action="store_true",
        help=(
            "Export only — do not open the interactive Tkinter window. "
            "Requires --save. Useful for pre-generating images before a demo."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=180,
        help="DPI for raster exports (PNG). Default: 180.",
    )
    return parser.parse_args()


def _export_headless(args):
    """
    Render the map without opening a Tkinter window and write to --save PATH.
    Uses the Agg (non-interactive) matplotlib backend.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # re-import after backend switch

    iso_scores, country_detail, file_count = (
        ({}, {}, 0) if args.demo
        else load_scores_from_csvs(args.analyst_outputs_dir)
    )
    if not iso_scores and not args.demo:
        print(
            f"  [!] No *_ddds_scores.csv files found in '{args.analyst_outputs_dir}'.\n"
            f"      Falling back to demo mode for the exported image."
        )
    apply_scores(iso_scores, demo_mode=(args.demo or not iso_scores))

    fig, ax = _plt.subplots(figsize=(16, 8), dpi=args.dpi, facecolor=BG_DARK)
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

    sm = _plt.cm.ScalarMappable(cmap=RISK_CMAP, norm=_plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.02, aspect=30)
    cbar.set_label(
        "MEAN RAW VAGUE_PROB", fontsize=8, color=TEXT_DIM,
        fontfamily="Consolas", labelpad=8,
    )
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
    cbar.ax.tick_params(labelsize=8, colors=TEXT_DIM, length=0)
    cbar.outline.set_edgecolor(BORDER_CLR)
    cbar.outline.set_linewidth(0.5)

    ax.add_patch(_plt.Rectangle((-178, -57), 3, 4, fc=GREY_ZERO, ec="none", zorder=5))
    ax.text(-174, -54.5, "0.0 — Unscored", fontsize=8, fontfamily="Consolas",
            color=TEXT_DIM, ha="left", va="center")
    ax.text(0.01, 0.99,
            "Score = mean raw vague_prob of sentences mentioning each geography",
            transform=ax.transAxes, fontsize=7, fontfamily="Consolas",
            color=TEXT_DIM, ha="left", va="top")

    mode_label = (
        "DEMO — synthetic scores" if (args.demo or not iso_scores)
        else f"Pipeline data: {args.analyst_outputs_dir}  ({file_count} filing(s))"
    )
    fig.text(0.01, 0.01,
             f"DDDS v2.2  ·  CFA AI Investment Challenge 2026  ·  {mode_label}",
             fontsize=7, fontfamily="Consolas", color=TEXT_DIM)

    if args.demo or not iso_scores:
        ax.text(0.5, 0.5, "DEMO MODE — SYNTHETIC SCORES",
                ha="center", va="center", fontsize=14,
                fontfamily="Consolas", color="#ea580c", alpha=0.18,
                transform=ax.transAxes, fontweight="bold")

    fig.tight_layout(pad=0.4)

    save_path = args.save
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor=BG_DARK)
    _plt.close(fig)
    print(f"  [OK] Heatmap saved -> {os.path.abspath(save_path)}")


if __name__ == "__main__":
    args = parse_args()

    if args.headless:
        if not args.save:
            print("  [!] --headless requires --save PATH.  Exiting.")
            sys.exit(1)
        _export_headless(args)
        sys.exit(0)

    # ── Interactive Tkinter window ───────────────────────────────────────────
    matplotlib.use("TkAgg")
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()

    # Dark title bar on Windows 10/11
    try:
        import ctypes
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int),
        )
    except Exception:
        pass

    app = DDDSApp(
        root,
        outputs_dir=args.analyst_outputs_dir,
        demo_mode=args.demo,
    )


    def _on_close():
        plt.close("all")
        root.quit()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", _on_close)



    # Optional auto-save after the window finishes its first draw
    if args.save:
        def _auto_save():
            save_path = args.save
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            app.fig.savefig(
                save_path, dpi=args.dpi, bbox_inches="tight", facecolor=BG_DARK,
            )
            print(f"  [OK] Heatmap saved -> {os.path.abspath(save_path)}")
        root.after(500, _auto_save)

    root.mainloop()
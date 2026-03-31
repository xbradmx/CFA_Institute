"""
DDDS — Disclosure Degradation Detection System
Region Heat Risk Map  (Desktop Application)

Bloomberg-inspired dark-mode global heat map.  Each country gets a
random risk score 0.0–1.0 (1 d.p.).  Opens as a native resizable
desktop window.

Requirements
------------
    pip install geopandas==0.14.4 matplotlib

Run
---
    python ddds_heat_map.py
"""

import random
import tkinter as tk
from tkinter import ttk

import geopandas as gpd
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["agg.path.chunksize"] = 10000  # smoother rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ── Palette ────────────────────────────────────────────────────────────────

GREY_ZERO   = "#2d3340"      # lighter grey for 0-score countries
BG_DARK     = "#0d1117"      # main background
BG_PANEL    = "#161b22"      # header / footer panels
BORDER_CLR  = "#30363d"      # ui separators
BORDER_MAP  = "#ffffff"      # country outlines
TEXT_PRIMARY = "#e0e4ec"
TEXT_DIM     = "#484f5a"
ACCENT       = "#58a6ff"

# Risk colour-map  (0.1 green → 1.0 red)
_stops = [
    (0.00, "#16a34a"),   # green
    (0.22, "#65a30d"),   # lime
    (0.44, "#ca8a04"),   # amber
    (0.67, "#ea580c"),   # orange
    (1.00, "#dc2626"),   # red
]


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


RISK_CMAP = LinearSegmentedColormap.from_list(
    "ddds", [(p, _hex(c)) for p, c in _stops]
)


def score_colour(s):
    """0 → grey, 0.1–1.0 → green…red."""
    if s == 0:
        return GREY_ZERO
    t = max(0.0, min(1.0, (s - 0.1) / 0.9))
    r, g, b, _ = RISK_CMAP(t)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ── Load map data ─────────────────────────────────────────────────────────

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = world[world["name"] != "Antarctica"].copy()
world.reset_index(drop=True, inplace=True)


def roll_scores():
    world["score"]  = [round(random.randint(0, 10) / 10, 1) for _ in range(len(world))]
    world["colour"] = world["score"].apply(score_colour)


roll_scores()


# Spatial index for fast hover lookup
from shapely.strtree import STRtree   # noqa: E402

_tree  = STRtree(world.geometry.values)
_geoms = list(world.geometry.values)


def country_at(x, y):
    """Return (name, score) for the country under (lon, lat), or None."""
    from shapely.geometry import Point
    pt = Point(x, y)
    hits = _tree.query(pt)
    for idx in hits:
        if _geoms[idx].contains(pt):
            row = world.iloc[idx]
            return row["name"], row["score"]
    return None


# ── Application ───────────────────────────────────────────────────────────

class DDDSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disclosure Degradation Detection System")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1360x780")
        self.root.minsize(900, 550)

        self._build_header()
        self._build_map()
        self._build_footer()
        self.draw_map()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._last_hover = None

    # ── UI pieces ──────────────────────────────────────────────────────

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

        self.count_lbl = tk.Label(
            right, text=f"{len(world)} regions",
            font=("Consolas", 9), fg=TEXT_DIM, bg=BG_PANEL)
        self.count_lbl.pack(side="left", padx=(0, 14))

        tk.Button(
            right, text="↻  Regenerate", font=("Helvetica", 9),
            fg=ACCENT, bg=BG_PANEL, activeforeground="#79c0ff",
            activebackground="#1c2128", bd=0, padx=12, pady=4,
            relief="flat", cursor="hand2", command=self._regenerate,
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

        # toolbar
        tb_frame = tk.Frame(self.root, bg=BG_PANEL)
        tb_frame.pack(fill="x", side="bottom")
        tb = NavigationToolbar2Tk(self.canvas, tb_frame)
        tb.config(bg=BG_PANEL)
        tb.update()
        for w in tb.winfo_children():
            try: w.config(bg=BG_PANEL, highlightbackground=BG_PANEL)
            except tk.TclError: pass

    def _build_footer(self):
        ft = tk.Frame(self.root, bg=BG_PANEL, height=26)
        ft.pack(fill="x", side="bottom")
        ft.pack_propagate(False)
        tk.Label(ft, text="DDDS v2.1  ·  CFA AI Investment Challenge 2026",
                 font=("Consolas", 8), fg=TEXT_DIM, bg=BG_PANEL
                 ).pack(side="right", padx=16)

    # ── Drawing ────────────────────────────────────────────────────────

    def draw_map(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal")

        # Filled countries
        world.plot(ax=ax, color=world["colour"], linewidth=0, antialiased=True)

        # Country borders — thicker, brighter
        world.boundary.plot(
            ax=ax,
            edgecolor=BORDER_MAP,
            linewidth=0.55,
            alpha=0.30,
            antialiased=True,
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        # Colour bar
        sm = plt.cm.ScalarMappable(cmap=RISK_CMAP,
                                   norm=plt.Normalize(0.1, 1.0))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.02,
                                 aspect=30)
        cbar.set_label("RISK SCORE", fontsize=8, color=TEXT_DIM,
                        fontfamily="Consolas", labelpad=8)
        cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 1.0])
        cbar.set_ticklabels(["0.1", "0.3", "0.5", "0.7", "1.0"])
        cbar.ax.tick_params(labelsize=8, colors=TEXT_DIM, length=0)
        cbar.outline.set_edgecolor(BORDER_CLR)
        cbar.outline.set_linewidth(0.5)

        # Zero legend
        ax.add_patch(plt.Rectangle((-178, -57), 3, 4,
                                    fc=GREY_ZERO, ec="none", zorder=5))
        ax.text(-174, -54.5, "0.0 — Unscored", fontsize=8,
                fontfamily="Consolas", color=TEXT_DIM, ha="left", va="center")

        # Tooltip annotation (re-created each draw)
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

    # ── Interactivity ──────────────────────────────────────────────────

    def _regenerate(self):
        roll_scores()
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
            name, score = result
            if self._last_hover == name:
                # just reposition, skip redraw text
                self.annot.xy = (event.xdata, event.ydata)
                self.canvas.draw_idle()
                return
            self._last_hover = name
            clr = score_colour(score) if score > 0 else TEXT_DIM
            self.annot.xy = (event.xdata, event.ydata)
            self.annot.set_text(f"{name}\nRisk Score: {score:.1f}")
            self.annot.set_color(clr)
            self.annot.set_visible(True)
        else:
            if self.annot.get_visible():
                self.annot.set_visible(False)
            self._last_hover = None

        self.canvas.draw_idle()


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()

    # Dark title bar on Windows 10/11
    try:
        import ctypes
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
    except Exception:
        pass

    app = DDDSApp(root)
    root.mainloop()
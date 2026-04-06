# =============================================================================
# DDDS — Desktop Analyst Interface
# Disclosure Degradation Detection System | The Transparency Project
# Lancaster University | CFA AI Investment Challenge 2026
#
# Run from the project root:  python ddds_app.py
# Requires analyst_pipeline.py to be in the same directory.
#
# pip install customtkinter
# =============================================================================

import os
import sys
import threading
import queue
import csv
from datetime import datetime
from tkinter import filedialog
import tkinter as tk

import customtkinter as ctk

# ── Resolve analyst_pipeline from same directory as this script ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "pipeline",
    os.path.join(_HERE, "RUN USER BACKEND - vague_complex_analysis_pipeline.py")
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

# =============================================================================
# Colour palette  (mirrors the HTML demo)
# =============================================================================
BG        = "#07090f"
SURFACE   = "#0e1118"
RAISED    = "#141820"
BORDER    = "#1e2535"
AMBER     = "#c9924a"
AMBER_LO  = "#1a1408"
TXT       = "#dde2ec"
TXT_2     = "#7a8aaa"
TXT_3     = "#3d4d63"
RED       = "#c45c5c"
RED_LO    = "#1a0c0c"
BLUE      = "#4d7fbe"
BLUE_LO   = "#0c1220"
GREEN     = "#4a9a6e"
GREEN_LO  = "#0c1a12"
WHITE     = "#ffffff"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# =============================================================================
# Worker — runs the pipeline in a background thread
# =============================================================================

class PipelineWorker:
    """
    Executes analyst_pipeline functions in a daemon thread.
    Posts status tuples to a Queue consumed by the UI.

    Queue message format:
      ("step_start",  step_num, detail_text)
      ("step_done",   step_num)
      ("step_error",  step_num, error_text)
      ("log",         text)
      ("results",     results_dict)
      ("fatal",       error_text)
    """

    def __init__(self, filepath: str, q: queue.Queue):
        self.filepath = filepath
        self.q = q
        self._models = None

    def _post(self, *args):
        self.q.put(args)

    def run(self):
        fp = self.filepath
        filename = os.path.basename(fp)

        try:
            # ── Step 1 — Parse document ──────────────────────────────────────
            self._post("step_start", 1, f"Reading {os.path.splitext(filename)[1].upper()} · detecting filing type")
            try:
                text, filing_type = pipeline.load_document(fp)
            except Exception as e:
                self._post("step_error", 1, str(e))
                return

            if filing_type == "UNKNOWN":
                self._post("log", "Warning: filing type not detected — using grand-mean baselines")
            self._post("step_done", 1)

            # ── Step 2 — Extract sections ────────────────────────────────────
            self._post("step_start", 2, "Locating MD&A and Risk Factors sections")
            sections = pipeline.extract_sections(text)
            section_names = list(sections.keys())
            self._post("log", f"Sections found: {', '.join(section_names)}")
            self._post("step_done", 2)

            # ── Step 3 — Segment + infer ─────────────────────────────────────
            all_sentences = []
            for section_name, section_text in sections.items():
                sents = pipeline.segment_sentences(section_text)
                for idx, sent in enumerate(sents):
                    all_sentences.append({
                        "sentence_id":    f"{os.path.splitext(filename)[0]}_{section_name}_{idx:04d}",
                        "section":        section_name,
                        "text":           sent,
                        "vague_prob":     None,
                        "vague_label":    None,
                        "complex_prob":   None,
                        "complex_label":  None,
                        "demeaned_vague":   None,
                        "demeaned_complex": None,
                        "outlier":          False,
                    })

            if not all_sentences:
                self._post("fatal", "No sentences extracted from document.")
                return

            self._post("step_start", 3,
                       f"{len(all_sentences)} sentences · loading FinBERT classifiers")

            # Load models if not already loaded
            if self._models is None:
                self._post("log", "Loading vagueness model …")
                vm  = pipeline.AutoModelForSequenceClassification.from_pretrained(pipeline.VAGUE_MODEL_PATH)
                vt  = pipeline.AutoTokenizer.from_pretrained(pipeline.VAGUE_MODEL_PATH)
                self._post("log", "Loading complexity model …")
                cm  = pipeline.AutoModelForSequenceClassification.from_pretrained(pipeline.COMPLEX_MODEL_PATH)
                ct  = pipeline.AutoTokenizer.from_pretrained(pipeline.COMPLEX_MODEL_PATH)
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                vm.to(device).eval()
                cm.to(device).eval()
                self._models = (vm, vt, cm, ct, device)
                self._post("log", f"Models loaded on {device}")

            vm, vt, cm, ct, device = self._models

            self._post("log", "Running vagueness inference …")
            texts = [r["text"] for r in all_sentences]
            vague_preds   = pipeline.run_inference(texts, vm, vt, device)

            self._post("log", "Running complexity inference …")
            complex_preds = pipeline.run_inference(texts, cm, ct, device)

            for i, row in enumerate(all_sentences):
                row["vague_prob"],   row["vague_label"]   = round(vague_preds[i][0], 4),   vague_preds[i][1]
                row["complex_prob"], row["complex_label"] = round(complex_preds[i][0], 4), complex_preds[i][1]

            self._post("step_done", 3)

            # ── Step 4 — Demean + outliers ────────────────────────────────────
            self._post("step_start", 4, f"Applying {filing_type} section-level baselines · 1.5 SD threshold")
            all_sentences = pipeline.demean_scores(all_sentences, filing_type)
            outliers      = pipeline.find_outliers(all_sentences)
            summary       = pipeline.build_summary(all_sentences, filing_type, filename)
            self._post("step_done", 4)

            # ── Write CSV ─────────────────────────────────────────────────────
            out_path = pipeline.write_csv(all_sentences, summary, fp, filing_type)
            self._post("log", f"CSV saved: {out_path}")

            self._post("results", {
                "filename":   filename,
                "filepath":   fp,
                "filing_type": filing_type,
                "summary":    summary,
                "outliers":   outliers,
                "total_sentences": len(all_sentences),
                "csv_path":   out_path,
            })

        except Exception as e:
            import traceback
            self._post("fatal", traceback.format_exc())


# =============================================================================
# Reusable UI components
# =============================================================================

def make_label(parent, text, size=13, color=TXT_2, weight="normal", anchor="w", **kw):
    return ctk.CTkLabel(
        parent, text=text, font=ctk.CTkFont(size=size, weight=weight),
        text_color=color, anchor=anchor, **kw
    )


def make_section_divider(parent, text):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(fill="x", pady=(20, 10))
    make_label(frame, text.upper(), size=10, color=TXT_3).pack(side="left")
    ctk.CTkFrame(frame, fg_color=BORDER, height=1).pack(
        side="left", fill="x", expand=True, padx=(12, 0), pady=6
    )
    return frame


class StepRow(ctk.CTkFrame):
    STATES = {
        "idle":    (TXT_3,  "—",         SURFACE),
        "active":  (AMBER,  "running…",  RAISED),
        "done":    (GREEN,  "done",       SURFACE),
        "error":   (RED,    "error",      RED_LO),
    }

    def __init__(self, parent, num: int, title: str, detail: str):
        super().__init__(parent, fg_color=SURFACE, corner_radius=0)
        self.pack(fill="x")
        ctk.CTkFrame(self, fg_color=BORDER, height=1).pack(fill="x")

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=14)

        self._num_lbl = make_label(inner, f"0{num}", size=12, color=TXT_3, weight="normal")
        self._num_lbl.pack(side="left", padx=(0, 16))

        info = ctk.CTkFrame(inner, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True)

        self._title_lbl  = make_label(info, title,  size=13, color=TXT_2, weight="normal")
        self._title_lbl.pack(anchor="w")
        self._detail_lbl = make_label(info, detail, size=11, color=TXT_3, weight="normal")
        self._detail_lbl.pack(anchor="w")

        self._status_lbl = make_label(inner, "—", size=11, color=TXT_3, anchor="e")
        self._status_lbl.pack(side="right")

        self._set_bg(SURFACE)

    def _set_bg(self, color):
        self.configure(fg_color=color)

    def set_state(self, state: str, detail: str = None):
        color, status_text, bg = self.STATES[state]
        self._num_lbl.configure(text_color=color)
        self._status_lbl.configure(text=status_text, text_color=color)
        self._set_bg(bg)
        if state == "active":
            self._title_lbl.configure(text_color=TXT)
            self._detail_lbl.configure(text_color=AMBER)
        else:
            self._title_lbl.configure(text_color=TXT_2)
            self._detail_lbl.configure(text_color=TXT_3)
        if detail:
            self._detail_lbl.configure(text=detail)


class ScoreBar(ctk.CTkFrame):
    """Horizontal score bar with baseline tick and delta badge."""

    def __init__(self, parent, label: str, raw: float, demeaned: float,
                 baseline: float, color: str, lo_color: str):
        super().__init__(parent, fg_color="transparent")
        self.pack(fill="x", pady=(0, 14))

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 6))
        make_label(header, label, size=12, color=TXT_2).pack(side="left")

        delta_sign = "+" if demeaned >= 0 else ""
        delta_txt  = f"{delta_sign}{demeaned:.3f}"
        delta_fg   = RED if demeaned >= 0 else GREEN
        delta_bg   = RED_LO if demeaned >= 0 else GREEN_LO

        val_frame = ctk.CTkFrame(header, fg_color="transparent")
        val_frame.pack(side="right")
        make_label(val_frame, f"{raw:.3f}", size=15, color=TXT, weight="bold").pack(side="left", padx=(0, 8))
        badge = ctk.CTkLabel(
            val_frame, text=delta_txt,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=delta_fg, fg_color=delta_bg,
            corner_radius=4, padx=6, pady=2
        )
        badge.pack(side="left")

        # Bar track
        track = ctk.CTkFrame(self, fg_color=RAISED, height=6, corner_radius=3)
        track.pack(fill="x")
        track.update_idletasks()
        track_w = track.winfo_width() or 400

        fill_w = max(4, int(raw * track_w))
        fill = ctk.CTkFrame(track, fg_color=color, height=6, width=fill_w, corner_radius=3)
        fill.place(x=0, y=0)

        # Baseline tick
        tick_x = int(baseline * track_w) - 1
        tick = ctk.CTkFrame(track, fg_color=TXT_3, width=2, height=12, corner_radius=1)
        tick.place(x=tick_x, y=-3)


class MetricCard(ctk.CTkFrame):
    def __init__(self, parent, label: str, value: str, value_color=TXT, sub: str = ""):
        super().__init__(parent, fg_color=SURFACE, corner_radius=8,
                         border_width=1, border_color=BORDER)
        make_label(self, label.upper(), size=10, color=TXT_3).pack(anchor="w", padx=14, pady=(12, 0))
        make_label(self, value, size=22, color=value_color, weight="bold").pack(anchor="w", padx=14, pady=(4, 0))
        if sub:
            make_label(self, sub, size=10, color=TXT_3).pack(anchor="w", padx=14, pady=(2, 10))
        else:
            ctk.CTkFrame(self, fg_color="transparent", height=10).pack()


# =============================================================================
# Main application
# =============================================================================

class DDDSApp(ctk.CTk):

    STEPS = [
        ("Parse document",          "Detect format and filing type"),
        ("Extract sections",        "MD&A · Risk Factors"),
        ("Run FinBERT inference",   "Vagueness · Complexity classifiers"),
        ("Compute demeaned scores", "Apply section-level baselines"),
    ]

    def __init__(self):
        super().__init__()

        self.title("DDDS — Disclosure Degradation Detection System")
        self.geometry("860x740")
        self.minsize(760, 620)
        self.configure(fg_color=BG)

        self._queue   = queue.Queue()
        self._worker  = None
        self._results = None

        self._build_header()
        self._build_body()
        self._build_footer()

        self._show_upload()
        self.after(100, self._poll_queue)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0,
                            border_width=0, height=58)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        inner = ctk.CTkFrame(hdr, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=32, pady=0)

        logo_frame = ctk.CTkFrame(inner, fg_color="transparent")
        logo_frame.pack(side="left", pady=0, fill="y")

        badge_frame = ctk.CTkFrame(
            logo_frame, fg_color="transparent",
            corner_radius=4, border_width=1, border_color=AMBER
        )
        badge_frame.pack(side="left", anchor="center")
        badge = ctk.CTkLabel(
            badge_frame, text="DDDS",
            font=ctk.CTkFont(family="Courier", size=12, weight="bold"),
            text_color=AMBER, fg_color="transparent",
            corner_radius=4
        )
        badge.pack(padx=8, pady=3)
        ctk.CTkFrame(logo_frame, fg_color="transparent", width=12).pack(side="left")
        make_label(logo_frame, "Disclosure Degradation Detection System",
                   size=14, color=TXT, weight="normal").pack(side="left", anchor="center")

        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.pack(side="right", fill="y")
        make_label(right, "The Transparency Project", size=11, color=TXT_3, anchor="e").pack(anchor="e")
        make_label(right, "Lancaster University · CFA 2026", size=11, color=TXT_3, anchor="e").pack(anchor="e")

        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(fill="x", side="top")

    def _build_body(self):
        self._body = ctk.CTkScrollableFrame(
            self, fg_color=BG, corner_radius=0,
            scrollbar_button_color=RAISED,
            scrollbar_button_hover_color=BORDER
        )
        self._body.pack(fill="both", expand=True, padx=0, pady=0)

        # Upload panel
        self._upload_panel = ctk.CTkFrame(self._body, fg_color="transparent")

        # Processing panel
        self._proc_panel = ctk.CTkFrame(self._body, fg_color="transparent")

        # Results panel
        self._results_panel = ctk.CTkFrame(self._body, fg_color="transparent")

        for p in (self._upload_panel, self._proc_panel, self._results_panel):
            p.pack(fill="both", expand=True, padx=48, pady=40)

        self._build_upload_panel()
        self._build_proc_panel()

    def _build_footer(self):
        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(fill="x", side="bottom")
        ftr = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=40)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)

        inner = ctk.CTkFrame(ftr, fg_color="transparent")
        inner.pack(fill="both", padx=32)

        make_label(inner, "DDDS · Analyst Inference Pipeline · Run 6 Demo",
                   size=11, color=TXT_3).pack(side="left", anchor="center", pady=10)
        make_label(inner, "FinBERT · Two-class · Vagueness + Complexity",
                   size=11, color=TXT_3, anchor="e").pack(side="right", anchor="center", pady=10)

    # ── Upload panel ──────────────────────────────────────────────────────────

    def _build_upload_panel(self):
        p = self._upload_panel

        ctk.CTkFrame(p, fg_color="transparent", height=20).pack()

        make_label(p, "ANALYST INFERENCE PIPELINE", size=10, color=AMBER, anchor="center"
                   ).pack(fill="x", pady=(0, 14))

        make_label(
            p, "Surface disclosure risk\nbefore it surfaces itself",
            size=30, color=TXT, weight="bold", anchor="center", justify="center"
        ).pack(fill="x", pady=(0, 12))

        make_label(
            p,
            "Upload a 10-K or 10-Q filing. The pipeline extracts MD&A and Risk Factor sections,\n"
            "scores each sentence for vagueness and complexity, and returns peer-demeaned signals.",
            size=13, color=TXT_2, anchor="center", justify="center"
        ).pack(fill="x", pady=(0, 36))

        # Drop zone
        dz = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=12,
                           border_width=1, border_color=BORDER)
        dz.pack(fill="x", pady=(0, 8))

        inner = ctk.CTkFrame(dz, fg_color="transparent")
        inner.pack(padx=40, pady=40)

        make_label(inner, "Select your filing", size=15, color=TXT,
                   weight="bold", anchor="center").pack(pady=(0, 6))
        make_label(inner, "Accepts .html · .pdf · .txt",
                   size=12, color=TXT_2, anchor="center").pack(pady=(0, 20))

        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack()

        self._btn_file = ctk.CTkButton(
            btn_row, text="Browse file", width=160, height=38,
            fg_color=RAISED, hover_color=BORDER,
            text_color=TXT, border_color=BORDER, border_width=1,
            font=ctk.CTkFont(size=13), command=self._browse_file
        )
        self._btn_file.pack(side="left", padx=(0, 10))

        self._btn_folder = ctk.CTkButton(
            btn_row, text="Browse folder", width=160, height=38,
            fg_color=RAISED, hover_color=BORDER,
            text_color=TXT_2, border_color=BORDER, border_width=1,
            font=ctk.CTkFont(size=13), command=self._browse_folder
        )
        self._btn_folder.pack(side="left")

        self._selected_lbl = make_label(inner, "", size=12, color=TXT_3, anchor="center")
        self._selected_lbl.pack(pady=(16, 0))

        self._btn_run = ctk.CTkButton(
            inner, text="Run analysis", width=340, height=44,
            fg_color=AMBER, hover_color="#a8792f",
            text_color="#07090f",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._run_pipeline,
            state="disabled"
        )
        self._btn_run.pack(pady=(16, 0))

        # Info strip
        strip = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=8,
                             border_width=1, border_color=BORDER)
        strip.pack(fill="x", pady=(20, 0))
        strip.columnconfigure((0, 1, 2), weight=1)

        cells = [
            ("Models",    "FinBERT · fine-tuned"),
            ("Baseline",  "Training set · demeaned"),
            ("Coverage",  "SIC 3400–3599"),
        ]
        for col, (label, val) in enumerate(cells):
            cell = ctk.CTkFrame(strip, fg_color="transparent")
            cell.grid(row=0, column=col, padx=24, pady=18, sticky="w")
            make_label(cell, label.upper(), size=10, color=TXT_3).pack(anchor="w")
            make_label(cell, val, size=13, color=TXT_2).pack(anchor="w", pady=(4, 0))

        self._selected_path = None

    # ── Processing panel ──────────────────────────────────────────────────────

    def _build_proc_panel(self):
        p = self._proc_panel

        hdr = ctk.CTkFrame(p, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 24))

        make_label(hdr, "PROCESSING", size=10, color=AMBER).pack(anchor="w")
        self._proc_filename_lbl = make_label(hdr, "—", size=24, color=TXT, weight="bold")
        self._proc_filename_lbl.pack(anchor="w", pady=(6, 0))

        # Steps
        steps_frame = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=10,
                                   border_width=1, border_color=BORDER)
        steps_frame.pack(fill="x", pady=(0, 20))

        self._step_rows = []
        for i, (title, detail) in enumerate(self.STEPS, 1):
            row = StepRow(steps_frame, i, title, detail)
            self._step_rows.append(row)

        # Log output
        make_label(p, "LOG", size=10, color=TXT_3).pack(anchor="w", pady=(0, 6))
        self._log_box = ctk.CTkTextbox(
            p, height=140, fg_color=SURFACE,
            text_color=TXT_3, border_color=BORDER, border_width=1,
            font=ctk.CTkFont(family="Courier", size=11),
            corner_radius=8, state="disabled"
        )
        self._log_box.pack(fill="x")

    def _log(self, text: str):
        self._log_box.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_box.insert("end", f"[{ts}]  {text}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    # ── Results panel ─────────────────────────────────────────────────────────

    def _build_results_panel(self, data: dict):
        p = self._results_panel
        for w in p.winfo_children():
            w.destroy()

        fn   = data["filename"]
        ft   = data["filing_type"]
        summ = data["summary"]
        outs = data["outliers"]
        tot  = data["total_sentences"]

        # Header
        hdr = ctk.CTkFrame(p, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 28))

        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True)
        make_label(left, "ANALYSIS COMPLETE", size=10, color=AMBER).pack(anchor="w")
        make_label(left, fn, size=22, color=TXT, weight="bold").pack(anchor="w", pady=(6, 2))
        make_label(left, f"{ft} · MD&A + Risk Factors · FinBERT · section-demeaned",
                   size=11, color=TXT_3).pack(anchor="w")

        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right", anchor="n", pady=4)

        ctk.CTkButton(
            right, text="Open CSV", width=120, height=36,
            fg_color=RAISED, hover_color=BORDER,
            text_color=TXT_2, border_color=BORDER, border_width=1,
            font=ctk.CTkFont(size=12),
            command=lambda: self._open_csv(data["csv_path"])
        ).pack(pady=(0, 8))

        ctk.CTkButton(
            right, text="New analysis", width=120, height=36,
            fg_color="transparent", hover_color=RAISED,
            text_color=TXT_3, border_color=BORDER, border_width=1,
            font=ctk.CTkFont(size=12),
            command=self._show_upload
        ).pack()

        # Metric cards
        cards_frame = ctk.CTkFrame(p, fg_color="transparent")
        cards_frame.pack(fill="x", pady=(0, 28))
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1)

        metrics = [
            ("Filing type",       ft,           AMBER,  ""),
            ("Sentences scored",  str(tot),      TXT,    ""),
            ("Outliers flagged",  str(len(outs)), RED,   "above 1.5 SD threshold"),
            ("Sections detected", str(len(summ)), TXT,   ""),
        ]
        for col, (lbl, val, col_c, sub) in enumerate(metrics):
            card = MetricCard(cards_frame, lbl, val, value_color=col_c, sub=sub)
            card.grid(row=0, column=col, padx=(0 if col == 0 else 8, 0), sticky="nsew")

        # Section breakdown
        make_section_divider(p, "Section breakdown — demeaned vs training baseline")

        sec_card = ctk.CTkFrame(p, fg_color=SURFACE, corner_radius=10,
                                border_width=1, border_color=BORDER)
        sec_card.pack(fill="x", pady=(0, 0))

        for si, (sec_name, vals) in enumerate(summ.items()):
            sec_inner = ctk.CTkFrame(sec_card, fg_color="transparent")
            sec_inner.pack(fill="x", padx=28, pady=20)

            sh = ctk.CTkFrame(sec_inner, fg_color="transparent")
            sh.pack(fill="x", pady=(0, 16))
            make_label(sh, sec_name, size=14, color=TXT, weight="bold").pack(side="left")
            make_label(sh, f"{vals['sentences']} sentences", size=11, color=TXT_3).pack(side="right")

            bar_row = ctk.CTkFrame(sec_inner, fg_color="transparent")
            bar_row.pack(fill="x")
            bar_row.columnconfigure((0, 1), weight=1)

            ft_key = ft if ft in pipeline.VAGUE_BASELINES else "UNKNOWN"
            v_base = pipeline.VAGUE_BASELINES.get((ft_key, sec_name),
                     pipeline.VAGUE_FALLBACK.get(ft_key, 0.48))
            c_base = pipeline.COMPLEX_BASELINES.get((ft_key, sec_name),
                     pipeline.COMPLEX_FALLBACK.get(ft_key, 0.13))

            left_col = ctk.CTkFrame(bar_row, fg_color="transparent")
            left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
            ScoreBar(left_col, "Vagueness",
                     vals["mean_vague_raw"], vals["mean_vague_demeaned"],
                     v_base, RED, RED_LO)

            right_col = ctk.CTkFrame(bar_row, fg_color="transparent")
            right_col.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
            ScoreBar(right_col, "Complexity",
                     vals["mean_complex_raw"], vals["mean_complex_demeaned"],
                     c_base, BLUE, BLUE_LO)

            if si < len(summ) - 1:
                ctk.CTkFrame(sec_card, fg_color=BORDER, height=1).pack(fill="x")

        # Flagged sentences
        if outs:
            make_section_divider(
                p, f"Flagged sentences — {len(outs)} above 1.5 SD threshold"
            )
            for outlier in outs:
                self._make_flagged_card(p, outlier)

        make_section_divider(p, "")

    def _make_flagged_card(self, parent, outlier: dict):
        card = ctk.CTkFrame(
            parent, fg_color=SURFACE, corner_radius=0,
            border_width=0
        )
        card.pack(fill="x", pady=(0, 10))

        left_accent = ctk.CTkFrame(card, fg_color=AMBER, width=3, corner_radius=0)
        left_accent.pack(side="left", fill="y")

        inner = ctk.CTkFrame(card, fg_color=RAISED, corner_radius=8)
        inner.pack(side="left", fill="both", expand=True, padx=(0, 0), pady=0)

        hdr = ctk.CTkFrame(inner, fg_color="transparent")
        hdr.pack(fill="x", padx=18, pady=(14, 10))

        make_label(hdr, outlier["section"].upper(), size=10, color=AMBER).pack(side="left")

        badge_row = ctk.CTkFrame(hdr, fg_color="transparent")
        badge_row.pack(side="right")

        vd = outlier["demeaned_vague"]
        cd = outlier["demeaned_complex"]
        vd_str = ("+" if vd >= 0 else "") + f"{vd:.3f}"
        cd_str = ("+" if cd >= 0 else "") + f"{cd:.3f}"

        for txt, bg, fg in [
            (f"Vague {outlier['vague_prob']:.3f}  Δ{vd_str}", RED_LO, RED),
            (f"Complex {outlier['complex_prob']:.3f}  Δ{cd_str}", BLUE_LO, BLUE),
        ]:
            ctk.CTkLabel(
                badge_row, text=txt,
                font=ctk.CTkFont(family="Courier", size=11),
                text_color=fg, fg_color=bg,
                corner_radius=4, padx=8, pady=3
            ).pack(side="left", padx=(6, 0))

        excerpt = outlier["text"][:280] + "…" if len(outlier["text"]) > 280 else outlier["text"]
        make_label(inner, f'"{excerpt}"', size=12, color=TXT_2,
                   anchor="w", justify="left", wraplength=680
                   ).pack(fill="x", padx=18, pady=(0, 14))

    # ── State transitions ─────────────────────────────────────────────────────

    def _show_upload(self):
        self._upload_panel.pack(fill="both", expand=True, padx=48, pady=40)
        self._proc_panel.pack_forget()
        self._results_panel.pack_forget()

    def _show_processing(self):
        self._upload_panel.pack_forget()
        self._proc_panel.pack(fill="both", expand=True, padx=48, pady=40)
        self._results_panel.pack_forget()

        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

        for row in self._step_rows:
            row.set_state("idle")

    def _show_results(self):
        self._upload_panel.pack_forget()
        self._proc_panel.pack_forget()
        self._results_panel.pack(fill="both", expand=True, padx=48, pady=40)

    # ── File / folder selection ───────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select filing",
            filetypes=[
                ("Supported filings", "*.html *.htm *.pdf *.txt"),
                ("HTML", "*.html *.htm"),
                ("PDF", "*.pdf"),
                ("Text", "*.txt"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._selected_path = path
            self._selected_lbl.configure(text=os.path.basename(path), text_color=AMBER)
            self._btn_run.configure(state="normal")

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select folder of filings")
        if path:
            files = pipeline.collect_files(path)
            if not files:
                self._selected_lbl.configure(
                    text="No supported files found in that folder.", text_color=RED
                )
                self._btn_run.configure(state="disabled")
                return
            self._selected_path = path
            self._selected_lbl.configure(
                text=f"{len(files)} file(s) found in {os.path.basename(path)}",
                text_color=AMBER
            )
            self._btn_run.configure(state="normal")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run_pipeline(self):
        if not self._selected_path:
            return

        files = pipeline.collect_files(self._selected_path)
        if not files:
            return

        # For demo simplicity, process the first file;
        # for folder runs, process sequentially (one results panel shown at end)
        filepath = files[0]
        self._proc_filename_lbl.configure(text=os.path.basename(filepath))
        self._show_processing()

        worker = PipelineWorker(filepath, self._queue)
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]

                if kind == "step_start":
                    _, num, detail = msg
                    self._step_rows[num - 1].set_state("active", detail)
                    self._log(f"Step {num}: {self.STEPS[num - 1][0]}")

                elif kind == "step_done":
                    _, num = msg
                    self._step_rows[num - 1].set_state("done")

                elif kind == "step_error":
                    _, num, err = msg
                    self._step_rows[num - 1].set_state("error", err[:80])
                    self._log(f"Error in step {num}: {err}")

                elif kind == "log":
                    _, text = msg
                    self._log(text)

                elif kind == "results":
                    _, data = msg
                    self._results = data
                    self._build_results_panel(data)
                    self._show_results()

                elif kind == "fatal":
                    _, err = msg
                    self._log(f"Fatal: {err}")

        except queue.Empty:
            pass

        self.after(80, self._poll_queue)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _open_csv(self, path: str):
        import subprocess, platform
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    app = DDDSApp()
    app.mainloop()
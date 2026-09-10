"""Tkinter desktop app: point it at a folder, stitch, look around."""
from __future__ import annotations

import json
import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import __version__
from .insp import borrow_imu, find_insp_files, parse_insp
from .stitch import Stitcher, StitchOptions, save_pano_jpeg
from .viewer import PanoViewer

OUTPUT_DIRNAME = "stitched"
CALIB_FILENAME = "calibration.json"
WIDTHS = ["2048", "3040", "4096", "6080", "8192"]


class App(tk.Tk):
    def __init__(self, folder: Optional[str] = None):
        super().__init__()
        self.title(f"Insta360 ONE X stitcher {__version__}")
        self.geometry("1280x800")
        self.minsize(960, 500)

        self.folder: Optional[Path] = None
        self.files: List[Path] = []
        self.stitcher = Stitcher()
        self.queue: "queue.Queue" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.jobs: List[Path] = []
        self.busy = False
        self.cancel_requested = False

        self._build_ui()
        self.after(100, self._poll_queue)
        if folder:
            self.open_folder(folder)

    # ----- UI ------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=(6, 4))
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open folder…", command=self.choose_folder).pack(side="left")
        self.folder_var = tk.StringVar(value="(no folder)")
        ttk.Label(top, textvariable=self.folder_var, width=28, anchor="w").pack(side="left", padx=8)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(top, text="Width").pack(side="left")
        self.width_var = tk.StringVar(value="6080")
        ttk.Combobox(top, textvariable=self.width_var, values=WIDTHS, width=6, state="readonly").pack(side="left", padx=(2, 8))
        self.parallax_var = tk.BooleanVar(value=True)
        self.level_var = tk.BooleanVar(value=True)
        self.refine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Parallax fix", variable=self.parallax_var).pack(side="left")
        ttk.Checkbutton(top, text="Auto level", variable=self.level_var).pack(side="left")
        ttk.Checkbutton(top, text="Refine lenses", variable=self.refine_var).pack(side="left")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        self.btn_one = ttk.Button(top, text="Stitch selected", command=self.stitch_selected)
        self.btn_one.pack(side="left")
        self.btn_all = ttk.Button(top, text="Stitch all", command=self.stitch_all)
        self.btn_all.pack(side="left", padx=4)
        self.btn_cancel = ttk.Button(top, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left")
        ttk.Button(top, text="Save view…", command=self.save_view).pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=(4, 4))
        body.add(left, weight=0)
        ttk.Label(left, text="Photos (.insp)").pack(anchor="w")
        lb_frame = ttk.Frame(left)
        lb_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(lb_frame, width=42, activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.show_selected())
        self.listbox.bind("<Double-Button-1>", lambda e: self.stitch_selected())
        self.info_text = tk.Text(left, height=9, width=42, wrap="word", state="disabled", font=("TkDefaultFont", 9))
        self.info_text.pack(fill="x", pady=(4, 0))

        self.viewer = PanoViewer(body)
        body.add(self.viewer, weight=1)

        bottom = ttk.Frame(self, padding=(6, 3))
        bottom.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=160)
        self.progress.pack(side="right")

    # ----- folder handling ------------------------------------------------
    def choose_folder(self):
        d = filedialog.askdirectory(title="Choose a folder containing .insp files")
        if d:
            self.open_folder(d)

    def open_folder(self, folder):
        self.folder = Path(folder)
        shown = str(self.folder)
        self.folder_var.set(shown if len(shown) <= 28 else "…" + shown[-27:])
        self.files = find_insp_files(self.folder)
        self._load_calibration_cache()
        self.refresh_list()
        if self.files:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(0)
            self.show_selected()
            self.set_status(f"{len(self.files)} .insp file(s) found")
        else:
            self.viewer.clear("No .insp files in this folder")
            self.set_status("No .insp files found")

    @property
    def out_dir(self) -> Path:
        return self.folder / OUTPUT_DIRNAME

    def output_for(self, src: Path) -> Path:
        return self.out_dir / (src.stem + ".jpg")

    def refresh_list(self):
        sel = self.listbox.curselection()
        self.listbox.delete(0, "end")
        for f in self.files:
            mark = "✓ " if self.output_for(f).exists() else "   "
            self.listbox.insert("end", f"{mark}{f.name}")
        for i in sel:
            if i < len(self.files):
                self.listbox.selection_set(i)

    def _load_calibration_cache(self):
        p = self.out_dir / CALIB_FILENAME
        if p.exists():
            try:
                self.stitcher.calib_cache.update(json.loads(p.read_text()))
            except Exception:
                pass

    def _save_calibration_cache(self):
        if self.folder is None or not self.stitcher.calib_cache:
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / CALIB_FILENAME).write_text(json.dumps(self.stitcher.calib_cache, indent=2))
        except Exception:
            pass

    # ----- viewing --------------------------------------------------------
    def selected_file(self) -> Optional[Path]:
        sel = self.listbox.curselection()
        if not sel or sel[0] >= len(self.files):
            return None
        return self.files[sel[0]]

    def show_selected(self):
        f = self.selected_file()
        if f is None:
            return
        out = self.output_for(f)
        if out.exists():
            pano = cv2.imread(str(out), cv2.IMREAD_COLOR)
            if pano is not None:
                self.viewer.set_pano(pano)
                self._show_info(out.with_suffix(".json"))
                self.set_status(f"Viewing {out.name}")
                return
        try:
            insp = parse_insp(f)
            raw = None
            if insp.thumbnail:
                raw = cv2.imdecode(np.frombuffer(insp.thumbnail, np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                raw = cv2.imdecode(np.frombuffer(insp.jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
            self.viewer.set_flat(raw, "Not stitched yet - double-click the file or press 'Stitch selected'")
            self._set_info(f"{f.name}\n{insp.model} {insp.serial}\nfirmware {insp.firmware}\n"
                           f"calibration: {'yes' if insp.calibration else 'missing'}\n"
                           f"IMU samples: {0 if insp.imu is None else len(insp.imu)}")
        except Exception as e:
            self.viewer.clear(f"Cannot read {f.name}: {e}")

    def _show_info(self, json_path: Path):
        if not json_path.exists():
            self._set_info("")
            return
        try:
            info = json.loads(json_path.read_text())
        except Exception:
            self._set_info("")
            return
        lines = [Path(info.get("file", "")).name]
        if "alignment" in info:
            lines.append(info["alignment"])
        lines.append("levelled from IMU" if info.get("levelled") else "not levelled (no IMU data or disabled)")
        for s in info.get("seams", []):
            if "flow_px_mean" in s:
                lines.append(f"seam {s['lon']:+.0f}: parallax flow mean {s['flow_px_mean']:.1f}px max {s['flow_px_max']:.0f}px")
        lines.append(f"stitched in {info.get('seconds', '?')} s, width {info.get('options', {}).get('width', '?')}")
        self._set_info("\n".join(lines))

    def _set_info(self, text: str):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def save_view(self):
        img = self.viewer.current_view_image()
        if img is None:
            messagebox.showinfo("Save view", "Stitch and open a panorama first.")
            return
        p = filedialog.asksaveasfilename(title="Save current view", defaultextension=".jpg",
                                         filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if p:
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.set_status(f"Saved {p}")

    # ----- stitching ------------------------------------------------------
    def options(self) -> StitchOptions:
        return StitchOptions(width=int(self.width_var.get()), parallax=self.parallax_var.get(),
                             level=self.level_var.get(), refine=self.refine_var.get())

    def stitch_selected(self):
        f = self.selected_file()
        if f is not None:
            self.enqueue([f])

    def stitch_all(self):
        self.enqueue([f for f in self.files if not self.output_for(f).exists()] or list(self.files))

    def enqueue(self, files: List[Path]):
        if self.busy:
            self.jobs.extend(files)
            self.set_status(f"Queued {len(files)} file(s)")
            return
        self.jobs = list(files)
        self._start_worker()

    def _start_worker(self):
        self.busy = True
        self.cancel_requested = False
        self.btn_one.configure(state="disabled")
        self.btn_all.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.progress.start(12)
        opts = self.options()
        jobs = self.jobs
        self.jobs = []

        def work():
            for src in jobs:
                if self.cancel_requested:
                    break
                try:
                    self.queue.put(("status", f"{src.name}: reading", None))
                    insp = parse_insp(src)
                    if opts.level and borrow_imu(insp, self.files):
                        self.queue.put(("status", f"{src.name}: using IMU data from a sibling shot", None))
                    pano, info = self.stitcher.stitch(insp, opts, progress=lambda s, n=src.name: self.queue.put(("status", f"{n}: {s}", None)))
                    out = self.output_for(src)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    save_pano_jpeg(out, pano, opts.jpeg_quality, insp.jpeg)
                    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
                    self.queue.put(("done", src, out))
                except Exception as e:
                    traceback.print_exc()
                    self.queue.put(("error", src, str(e)))
            self.queue.put(("finished", None, None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def cancel(self):
        self.cancel_requested = True
        self.jobs = []
        self.set_status("Cancelling after the current file…")

    def _poll_queue(self):
        try:
            while True:
                kind, a, b = self.queue.get_nowait()
                if kind == "status":
                    self.set_status(a)
                elif kind == "done":
                    self.refresh_list()
                    self._save_calibration_cache()
                    if self.selected_file() == a:
                        self.show_selected()
                    self.set_status(f"{a.name}: stitched -> {b}")
                elif kind == "error":
                    self.set_status(f"{a.name}: failed: {b}")
                elif kind == "finished":
                    self.busy = False
                    self.btn_one.configure(state="normal")
                    self.btn_all.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    self.progress.stop()
                    if self.jobs:
                        self._start_worker()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def set_status(self, text: str):
        self.status_var.set(text)


def main(folder: Optional[str] = None):
    app = App(folder)
    app.mainloop()

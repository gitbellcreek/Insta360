"""Interactive 360 viewer widget (Tkinter).

Drag to look around, scroll to zoom, arrow keys to pan, `F` to toggle
between the interactive rectilinear view and the flat equirectangular
image.  Rendering is a cv2.remap of the panorama through a pinhole camera.
"""
from __future__ import annotations

import math
import tkinter as tk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .geometry import dirs_to_equirect, rot_x, rot_y


class PanoViewer(tk.Canvas):
    MAX_PANO_WIDTH = 4096   # working copy size for interactive rendering

    def __init__(self, master, **kw):
        kw.setdefault("bg", "#202020")
        kw.setdefault("highlightthickness", 0)
        super().__init__(master, **kw)
        self.pano: Optional[np.ndarray] = None   # RGB
        self.flat: Optional[np.ndarray] = None   # RGB, shown when not a pano
        self.mode = "pano"                       # 'pano' | 'flat'
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 90.0
        self.message = "Open a folder with .insp files to begin"
        self._photo = None
        self._image_id = None
        self._drag = None
        self._render_job = None
        self._quality = 1.0
        self._grid_cache = {}

        self.bind("<Configure>", lambda e: self.schedule_render())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)          # Windows / macOS
        self.bind("<Button-4>", lambda e: self._zoom(-1))  # X11
        self.bind("<Button-5>", lambda e: self._zoom(+1))
        self.bind("<Double-Button-1>", lambda e: self.reset_view())
        self.bind("<Left>", lambda e: self.nudge(-5, 0))
        self.bind("<Right>", lambda e: self.nudge(5, 0))
        self.bind("<Up>", lambda e: self.nudge(0, 5))
        self.bind("<Down>", lambda e: self.nudge(0, -5))
        self.bind("<Key-f>", lambda e: self.toggle_mode())
        self.bind("<Key-F>", lambda e: self.toggle_mode())
        self.bind("<Enter>", lambda e: self.focus_set())

    # ----- content -----------------------------------------------------
    def set_pano(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] > self.MAX_PANO_WIDTH:
            rgb = cv2.resize(rgb, (self.MAX_PANO_WIDTH, self.MAX_PANO_WIDTH // 2), interpolation=cv2.INTER_AREA)
        self.pano = rgb
        self.flat = None
        self.mode = "pano"
        self.message = ""
        self.schedule_render()

    def set_flat(self, bgr: np.ndarray, message: str = ""):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] > 2048:
            s = 2048 / rgb.shape[1]
            rgb = cv2.resize(rgb, (2048, int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        self.flat = rgb
        self.pano = None
        self.mode = "flat"
        self.message = message
        self.schedule_render()

    def clear(self, message: str = ""):
        self.pano = None
        self.flat = None
        self.message = message
        self.schedule_render()

    def toggle_mode(self):
        if self.pano is None:
            return
        self.mode = "flat" if self.mode == "pano" else "pano"
        self.schedule_render()

    def reset_view(self):
        self.yaw, self.pitch, self.fov = 0.0, 0.0, 90.0
        self.schedule_render()

    def nudge(self, dyaw, dpitch):
        self.yaw = (self.yaw + dyaw + 180) % 360 - 180
        self.pitch = max(-90, min(90, self.pitch + dpitch))
        self.schedule_render()

    # ----- interaction -------------------------------------------------
    def _on_press(self, e):
        self.focus_set()
        self._drag = (e.x, e.y, self.yaw, self.pitch)

    def _on_drag(self, e):
        if self._drag is None or self.mode != "pano" or self.pano is None:
            return
        x0, y0, yaw0, pitch0 = self._drag
        w = max(1, self.winfo_width())
        deg_per_px = self.fov / w
        self.yaw = (yaw0 - (e.x - x0) * deg_per_px + 180) % 360 - 180
        self.pitch = max(-90.0, min(90.0, pitch0 + (e.y - y0) * deg_per_px))
        self._quality = 0.5
        self.schedule_render(delay=0)

    def _on_release(self, e):
        self._drag = None
        self._quality = 1.0
        self.schedule_render(delay=0)

    def _on_wheel(self, e):
        self._zoom(-1 if e.delta > 0 else 1)

    def _zoom(self, direction):
        if self.mode != "pano":
            return
        self.fov = max(20.0, min(140.0, self.fov * (1.1 ** direction)))
        self.schedule_render()

    # ----- rendering ---------------------------------------------------
    def schedule_render(self, delay: int = 15):
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(delay, self.render)

    def _grid(self, w: int, h: int, fov: float):
        key = (w, h, round(fov, 2))
        g = self._grid_cache.get(key)
        if g is None:
            f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
            xs = (np.arange(w, dtype=np.float32) - w / 2.0 + 0.5) / f
            ys = -(np.arange(h, dtype=np.float32) - h / 2.0 + 0.5) / f
            X, Y = np.meshgrid(xs, ys)
            d = np.stack([X, Y, np.ones_like(X)], axis=-1)
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            self._grid_cache = {key: d}
            g = d
        return g

    def render(self):
        self._render_job = None
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        img = None
        if self.mode == "pano" and self.pano is not None:
            q = self._quality
            rw, rh = max(2, int(w * q)), max(2, int(h * q))
            d = self._grid(rw, rh, self.fov)
            R = (rot_y(self.yaw) @ rot_x(-self.pitch)).astype(np.float32)
            dd = d @ R.T
            u, v = dirs_to_equirect(dd, self.pano.shape[1], self.pano.shape[0])
            view = cv2.remap(self.pano, u.astype(np.float32), v.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            if q != 1.0:
                view = cv2.resize(view, (w, h), interpolation=cv2.INTER_LINEAR)
            img = view
        else:
            src = self.flat if self.flat is not None else self.pano
            if src is not None:
                s = min(w / src.shape[1], h / src.shape[0])
                tw, th = max(1, int(src.shape[1] * s)), max(1, int(src.shape[0] * s))
                img = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
        if img is not None:
            self._photo = ImageTk.PhotoImage(Image.fromarray(img))
            self._image_id = self.create_image(w // 2, h // 2, image=self._photo, anchor="center")
        if self.message:
            self.create_text(w // 2, h // 2 if img is None else h - 24, text=self.message,
                             fill="#dddddd", font=("TkDefaultFont", 12), justify="center")
        if self.mode == "pano" and self.pano is not None:
            hud = f"yaw {self.yaw:+.0f}  pitch {self.pitch:+.0f}  fov {self.fov:.0f}   drag: look  wheel: zoom  F: flat view  double-click: reset"
            self.create_text(8, h - 8, text=hud, fill="#bbbbbb", anchor="sw", font=("TkDefaultFont", 9))

    def current_view_image(self) -> Optional[np.ndarray]:
        """The rectilinear view currently on screen as a BGR array (full quality)."""
        if self.mode != "pano" or self.pano is None:
            return None
        w, h = self.winfo_width(), self.winfo_height()
        d = self._grid(w, h, self.fov)
        R = (rot_y(self.yaw) @ rot_x(-self.pitch)).astype(np.float32)
        u, v = dirs_to_equirect(d @ R.T, self.pano.shape[1], self.pano.shape[0])
        view = cv2.remap(self.pano, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

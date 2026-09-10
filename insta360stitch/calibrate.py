"""Hugin-style alignment: find control points in the overlap zones and
optimise the lens model with them.

Free parameters (like Hugin's "v" and "y,p,r" of the second image):
  fov     field of view at the calibrated circle radius (shared: the factory
          radii already encode the per-lens scale difference)
  yaw2, pitch2, roll2   orientation of the rear lens relative to the front

A single lens pair cannot separate the FOV of the two lenses, nor the
polynomial distortion terms (only the rim is ever observed), so those are
left fixed - exactly as one would do in Hugin with two fisheye images.

Parallax is real signal here: nearby objects legitimately disagree between
the lenses.  A Cauchy robust loss lets the far/consistent points dominate
while the local optical-flow pass in stitch.py deals with the rest.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import DEG, Lens, equirect_dirs, lon_to_u, render_lens


@dataclass
class CalibrationResult:
    lenses: List[Lens]
    n_points: int
    rms_deg: float
    accepted: bool
    message: str


def _keypoints(gray: np.ndarray, mask: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.03)
    kp, des = sift.detectAndCompute(gray, mask.astype(np.uint8) * 255)
    return kp, des


def find_control_points(img: np.ndarray, lenses: List[Lens], render_w: int = 2048,
                        band_half_deg: float = 20.0, max_disp_frac: float = 0.03) -> np.ndarray:
    """Return an (N, 4) array of matched stored-frame pixels (X1, Y1, X2, Y2)."""
    render_h = render_w // 2
    I = np.eye(3)
    obs = []
    for lon_c in (-90.0, 90.0):
        u0 = int(lon_to_u(lon_c - band_half_deg, render_w))
        u1 = int(lon_to_u(lon_c + band_half_deg, render_w))
        A, va = render_lens(img, lenses[0], I, render_w, render_h, u_range=(u0, u1))
        B, vb = render_lens(img, lenses[1], I, render_w, render_h, u_range=(u0, u1))
        overlap = va & vb
        if overlap.sum() < 500:
            continue
        overlap = cv2.erode(overlap.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
        kpa, da = _keypoints(ga, overlap)
        kpb, db = _keypoints(gb, overlap)
        if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(da, db, k=2)
        pa, pb = [], []
        max_disp = max_disp_frac * render_w
        for m in knn:
            if len(m) < 2 or m[0].distance > 0.8 * m[1].distance:
                continue
            a = kpa[m[0].queryIdx].pt
            b = kpb[m[0].trainIdx].pt
            if abs(a[0] - b[0]) > max_disp or abs(a[1] - b[1]) > max_disp:
                continue
            pa.append(a)
            pb.append(b)
        if len(pa) < 4:
            continue
        pa = np.array(pa, np.float64)
        pb = np.array(pb, np.float64)
        # reject gross outliers against the median displacement of this band
        disp = pb - pa
        med = np.median(disp, axis=0)
        mad = np.median(np.abs(disp - med), axis=0) * 1.4826 + 2.0
        keep = np.all(np.abs(disp - med) < 6.0 * mad, axis=1)
        pa, pb = pa[keep], pb[keep]
        # render pixel -> world direction -> stored fisheye pixel
        def to_stored(pts, lens):
            u = pts[:, 0] + u0
            v = pts[:, 1]
            lon = ((u + 0.5) / render_w * 2.0 - 1.0) * math.pi
            lat = (0.5 - (v + 0.5) / render_h) * math.pi
            d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
            vl = d @ lens.M().T
            X, Y, _ = lens.project(vl)
            return np.stack([X, Y], -1)
        s1 = to_stored(pa, lenses[0])
        s2 = to_stored(pb, lenses[1])
        obs.append(np.hstack([s1, s2]))
    if not obs:
        return np.zeros((0, 4))
    return np.vstack(obs)


def _residuals(params: np.ndarray, base: List[Lens], obs: np.ndarray) -> np.ndarray:
    l1 = copy.copy(base[0])
    l2 = copy.copy(base[1])
    l1.fov = l2.fov = float(params[0])
    l2.yaw, l2.pitch, l2.roll = (float(x) for x in params[1:4])
    d1 = l1.unproject(obs[:, 0], obs[:, 1]) @ l1.M()
    d2 = l2.unproject(obs[:, 2], obs[:, 3]) @ l2.M()
    return ((d1 - d2) / DEG).ravel()


def optimise(lenses: List[Lens], obs: np.ndarray, iterations: int = 25, cauchy_deg: float = 1.0) -> Tuple[List[Lens], float]:
    """Levenberg-Marquardt with a Cauchy robust loss on angular residuals (deg)."""
    p = np.array([lenses[0].fov, lenses[1].yaw, lenses[1].pitch, lenses[1].roll], np.float64)
    steps = np.array([0.05, 0.02, 0.02, 0.02])
    lam = 1e-2

    def robust_cost(r):
        r2 = (r.reshape(-1, 3) ** 2).sum(1)
        return float(np.sum(np.log1p(r2 / cauchy_deg ** 2)))

    r = _residuals(p, lenses, obs)
    cost = robust_cost(r)
    for _ in range(iterations):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = np.zeros_like(p)
            dp[j] = steps[j]
            J[:, j] = (_residuals(p + dp, lenses, obs) - r) / steps[j]
        r2 = (r.reshape(-1, 3) ** 2).sum(1)
        w = np.repeat(1.0 / (1.0 + r2 / cauchy_deg ** 2), 3)
        JtW = J.T * w
        A = JtW @ J
        g = JtW @ r
        improved = False
        for _try in range(8):
            delta = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            p_new = p + delta
            r_new = _residuals(p_new, lenses, obs)
            c_new = robust_cost(r_new)
            if c_new < cost:
                p, r, cost = p_new, r_new, c_new
                lam = max(lam / 3.0, 1e-6)
                improved = True
                break
            lam *= 5.0
        if not improved or np.max(np.abs(delta)) < 1e-5:
            break
    out = [copy.copy(lenses[0]), copy.copy(lenses[1])]
    out[0].fov = out[1].fov = float(p[0])
    out[1].yaw, out[1].pitch, out[1].roll = (float(x) for x in p[1:4])
    return out, robust_rms(r, cauchy_deg)


def robust_rms(residuals: np.ndarray, cauchy_deg: float = 1.0) -> float:
    """RMS angular error (deg) over the inliers (< 3 * cauchy scale)."""
    r2 = (residuals.reshape(-1, 3) ** 2).sum(1)
    inl = r2 < (3 * cauchy_deg) ** 2
    return math.sqrt(float(r2[inl].mean())) if inl.any() else float("nan")


def calibrate(img: np.ndarray, lenses: List[Lens], rounds: int = 2, render_w: int = 2048,
              min_points: int = 20, fov_bounds=(170.0, 240.0), max_rot_change: float = 15.0,
              log=None) -> CalibrationResult:
    """Refine the lens model from the image content.

    The solution is accepted when it lowers the robust residual of the control
    points compared with the starting model and stays within sane bounds;
    otherwise the input lenses are returned unchanged.
    """
    cur = [copy.copy(l) for l in lenses]
    n = 0
    rms = float("nan")
    rms_before = float("nan")
    for rnd in range(rounds):
        obs = find_control_points(img, cur, render_w=render_w)
        n = len(obs)
        if log:
            log(f"alignment round {rnd + 1}: {n} control points")
        if n < min_points:
            return CalibrationResult(lenses, n, rms, False, f"only {n} control points, keeping previous lens model")
        if rnd == 0:
            p0 = np.array([lenses[0].fov, lenses[1].yaw, lenses[1].pitch, lenses[1].roll])
            rms_before = robust_rms(_residuals(p0, lenses, obs))
        cur, rms = optimise(cur, obs)
    fov_ok = fov_bounds[0] <= cur[0].fov <= fov_bounds[1]
    rot_ok = all(abs(getattr(cur[1], k) - getattr(lenses[1], k)) <= max_rot_change for k in ("yaw", "pitch", "roll"))
    better = not (rms_before == rms_before) or rms <= rms_before * 1.05 + 0.05
    if not (fov_ok and rot_ok and better):
        return CalibrationResult(lenses, n, rms, False,
                                 f"rejected implausible solution (fov {cur[0].fov:.1f}, rms {rms:.2f} deg, was {rms_before:.2f})")
    msg = (f"fov {cur[0].fov:.2f} deg, rear lens ypr ({cur[1].yaw:.2f}, {cur[1].pitch:.2f}, {cur[1].roll:.2f}), "
           f"rms {rms:.3f} deg (was {rms_before:.2f}) from {n} points")
    return CalibrationResult(cur, n, rms, True, msg)

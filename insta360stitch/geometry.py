"""Camera geometry: rotations, equirectangular grids and the fisheye lens model.

World frame: x right, y up, z forward (the optical axis of lens 1).
Equirectangular output: longitude -180..180 left to right (0 at the centre),
latitude +90 at the top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

DEG = math.pi / 180.0


def rot_x(deg: float) -> np.ndarray:
    a = deg * DEG
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = deg * DEG
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = deg * DEG
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation R with R @ a = b for unit vectors a, b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: pick any axis orthogonal to a
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def equirect_dirs(width: int, height: int, y0: int = 0, y1: int | None = None, lon_offset_deg: float = 0.0) -> np.ndarray:
    """Unit direction vectors for equirect pixels rows y0..y1, shape (rows, width, 3)."""
    if y1 is None:
        y1 = height
    lon = ((np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0) * np.pi + lon_offset_deg * DEG
    lat = (0.5 - (np.arange(y0, y1, dtype=np.float32) + 0.5) / height) * np.pi
    cl = np.cos(lat)[:, None]
    sl = np.sin(lat)[:, None]
    d = np.empty((y1 - y0, width, 3), np.float32)
    d[..., 0] = cl * np.sin(lon)[None, :]
    d[..., 1] = np.broadcast_to(sl, (y1 - y0, width))
    d[..., 2] = cl * np.cos(lon)[None, :]
    return d


def dirs_to_equirect(d: np.ndarray, width: int, height: int):
    """Inverse of equirect_dirs: directions (...,3) -> pixel (u, v) float arrays."""
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = (lon / np.pi + 1.0) * 0.5 * width - 0.5
    v = (0.5 - lat / np.pi) * height - 0.5
    return u, v


def lon_to_u(lon_deg: float, width: int) -> float:
    return (lon_deg / 180.0 + 1.0) * 0.5 * width


@dataclass
class Lens:
    """Fisheye lens model in the stored JPEG frame.

    Radial model r(theta) = f * (theta + k1 theta^3 + k2 theta^5) with f
    chosen so that r(fov/2) == r_cal, i.e. `fov` is the field of view at the
    calibrated circle radius.  The image-plane orientation of the lens is
    such that stored X grows with lens-frame y and stored Y with lens-frame
    x (this is how the ONE X stores its sensor, rotated by 90 degrees).

    `rotation` is (yaw, pitch, roll) in degrees; `flip` is True for the rear
    lens which looks along -z.  M() returns the world->lens rotation.
    """

    cx: float
    cy: float
    r_cal: float
    fov: float = 200.0
    k1: float = 0.0
    k2: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    flip: bool = False
    edge_frac: float = 0.985

    # ---- radial model -------------------------------------------------
    def _g(self, theta):
        t2 = theta * theta
        return theta * (1.0 + t2 * (self.k1 + t2 * self.k2))

    def _dg(self, theta):
        t2 = theta * theta
        return 1.0 + t2 * (3.0 * self.k1 + 5.0 * self.k2 * t2)

    @property
    def f(self) -> float:
        return self.r_cal / float(self._g(self.fov * 0.5 * DEG))

    def theta_of_r(self, r):
        """Invert the radial model (Newton), r in pixels -> theta in radians."""
        x = np.asarray(r, np.float64) / self.f
        theta = x.copy()
        for _ in range(6):
            theta = theta - (self._g(theta) - x) / self._dg(theta)
        return theta

    @property
    def theta_max(self) -> float:
        return float(self.theta_of_r(self.edge_frac * self.r_cal))

    def M(self) -> np.ndarray:
        R = euler_ypr(self.yaw, self.pitch, self.roll)
        if self.flip:
            R = R @ rot_y(180.0)
        return R.T

    # ---- projection ---------------------------------------------------
    def project(self, v: np.ndarray):
        """Lens-frame unit vectors (...,3) -> (X, Y, theta) in the stored image."""
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        theta = np.arccos(np.clip(vz, -1.0, 1.0))
        rho = np.hypot(vx, vy)
        rho = np.where(rho < 1e-9, 1e-9, rho)
        r = self.f * self._g(theta)
        X = self.cx + r * vy / rho
        Y = self.cy + r * vx / rho
        return X, Y, theta

    def unproject(self, X, Y) -> np.ndarray:
        """Stored image pixels -> lens-frame unit vectors (...,3)."""
        dx = np.asarray(X, np.float64) - self.cx
        dy = np.asarray(Y, np.float64) - self.cy
        r = np.hypot(dx, dy)
        theta = self.theta_of_r(r)
        s = np.sin(theta) / np.where(r < 1e-9, 1e-9, r)
        v = np.stack([dy * s, dx * s, np.cos(theta)], axis=-1)
        return v

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("cx", "cy", "r_cal", "fov", "k1", "k2", "yaw", "pitch", "roll", "flip", "edge_frac")}

    @classmethod
    def from_dict(cls, d: dict) -> "Lens":
        return cls(**{k: d[k] for k in ("cx", "cy", "r_cal", "fov", "k1", "k2", "yaw", "pitch", "roll", "flip", "edge_frac") if k in d})


def lenses_from_calibration(cal, img_w: int, img_h: int, fov: float = 200.0):
    """Build the two Lens objects from a parsed factory calibration."""
    centres = cal.stored_centres(img_w, img_h)
    lenses = []
    for i, (lc, (X, Y, r)) in enumerate(zip(cal.lenses, centres)):
        lenses.append(Lens(cx=X, cy=Y, r_cal=r, fov=fov, yaw=lc.yaw, pitch=lc.pitch, roll=lc.roll, flip=(i == 1)))
    return lenses


def render_lens(img: np.ndarray, lens: Lens, R_world: np.ndarray, width: int, height: int,
                chunk: int = 256, u_range=None):
    """Remap one fisheye image to equirectangular.

    R_world is an extra world rotation (levelling); the total world->lens
    rotation is lens.M() @ R_world.  Returns (image, valid_mask).
    u_range (u0, u1) limits the rendered columns; the returned arrays then
    have width u1-u0.
    """
    u0, u1 = (0, width) if u_range is None else u_range
    W = u1 - u0
    out = np.zeros((height, W, 3), np.uint8)
    valid = np.zeros((height, W), bool)
    RT = (lens.M() @ R_world).T.astype(np.float32)
    theta_max = lens.theta_max
    h_img, w_img = img.shape[:2]
    margin = 1.5
    for y0 in range(0, height, chunk):
        y1 = min(height, y0 + chunk)
        d = equirect_dirs(width, height, y0, y1)[:, u0:u1]
        v = d @ RT
        X, Y, theta = lens.project(v)
        ok = (theta < theta_max) & (X > margin) & (X < w_img - 1 - margin) & (Y > margin) & (Y < h_img - 1 - margin)
        mapx = np.where(ok, X, -1e4).astype(np.float32)
        mapy = np.where(ok, Y, -1e4).astype(np.float32)
        out[y0:y1] = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid[y0:y1] = ok
    return out, valid


def level_rotation(up_cam: np.ndarray) -> np.ndarray:
    """World rotation R such that camera-frame `up_cam` becomes world +y.

    render_lens applies v = M @ R @ d, so we need R with R @ (0,1,0) = up_cam.
    """
    return rotation_between(np.array([0.0, 1.0, 0.0]), up_cam)

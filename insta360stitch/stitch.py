"""The stitching pipeline.

    factory lens model  ->  (optional) control-point refinement
                        ->  levelling from the accelerometer
                        ->  render both lenses to equirectangular
                        ->  exposure gain match in the overlap
                        ->  per seam: seam finder (min-cost path, like enblend)
                                      optical-flow parallax warp toward the seam
                                      second seam pass on the warped images
                                      Laplacian pyramid blend
                        ->  JPEG with GPano/XMP 360 metadata

Why the flow pass: the two lenses of a ONE X are ~2.5 cm apart.  Anything
closer than a few metres appears in a different place in each lens, and no
single projection can align both the near and the far content.  The flow
warp pulls the two images into agreement locally in a band around the seam
only, tapering to zero away from it so the rest of the sphere keeps its
true geometry.
"""
from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibrate import calibrate
from .geometry import Lens, dirs_to_equirect, equirect_dirs, level_rotation, lenses_from_calibration, lon_to_u, render_lens, rot_y
from .insp import InspFile

Progress = Optional[Callable[[str], None]]


@dataclass
class StitchOptions:
    width: int = 6080            # output width; height is width // 2
    parallax: bool = True        # optical-flow parallax compensation in the seams
    level: bool = True           # use the accelerometer to level the horizon
    refine: bool = True          # refine the lens model from control points
    band_half_deg: float = 15.0  # half width of the seam processing band
    blend_levels: int = 5        # Laplacian pyramid depth
    jpeg_quality: int = 95
    yaw_offset: float = 0.0      # rotate the finished panorama (degrees)
    default_fov: float = 200.0   # starting FOV when nothing better is known

    def to_dict(self):
        return asdict(self)


def up_from_accel(accel: np.ndarray) -> np.ndarray:
    """ONE X accelerometer (sensor frame, in g) -> camera-frame up vector in
    our world axes.  Determined empirically: gravity sits on sensor -x when
    the camera is upright, and the sensor y axis maps to the world z axis."""
    a = np.asarray(accel, float)
    return np.array([a[2], -a[0], a[1]])


# ----------------------------------------------------------------------------
# seam finding
# ----------------------------------------------------------------------------

def seam_cost(A: np.ndarray, B: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(A, (0, 0), 1.5).astype(np.float32)
    b = cv2.GaussianBlur(B, (0, 0), 1.5).astype(np.float32)
    diff = np.abs(a - b).mean(axis=2)
    ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3) - cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)) * 0.125
    cost = diff + grad + 0.5
    cost[~overlap] = 1e4
    return cost


def min_cost_seam(cost: np.ndarray) -> np.ndarray:
    """Top-to-bottom minimum cost path, one column per row, moving at most
    one column per row.  Returns the column index per row."""
    H, W = cost.shape
    acc = cost.astype(np.float64).copy()
    back = np.zeros((H, W), np.int8)
    inf = np.inf
    for v in range(1, H):
        prev = acc[v - 1]
        left = np.empty(W)
        left[0] = inf
        left[1:] = prev[:-1]
        right = np.empty(W)
        right[-1] = inf
        right[:-1] = prev[1:]
        stacked = np.stack([left, prev, right])
        arg = stacked.argmin(axis=0)
        acc[v] = cost[v] + stacked[arg, np.arange(W)]
        back[v] = arg - 1
    seam = np.empty(H, np.int32)
    u = int(acc[-1].argmin())
    for v in range(H - 1, -1, -1):
        seam[v] = u
        u = u + int(back[v, u])
    return seam


# ----------------------------------------------------------------------------
# parallax compensation
# ----------------------------------------------------------------------------

def _fill_invalid(A: np.ndarray, B: np.ndarray, va: np.ndarray, vb: np.ndarray):
    Af = A.copy()
    Bf = B.copy()
    Af[~va] = B[~va]
    Bf[~vb] = A[~vb]
    return Af, Bf


def flow_warp(A: np.ndarray, B: np.ndarray, va: np.ndarray, vb: np.ndarray, seam: np.ndarray,
              taper_px: float, max_flow_frac: float = 0.5, fb_tol_px: float = 3.0
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp A and B half way toward each other around the seam.

    Dense optical flow (DIS) is computed both ways; where the forward and
    backward flows disagree (occlusions, the hand holding the camera, moving
    objects) the flow is discarded so those areas fall back to the plain
    seam.  The remaining flow is applied with a weight that is 1 on the seam
    and fades to 0 at `taper_px` from it, restricted to the overlap zone.

    Returns (A_warped, B_warped, flow_magnitude_map).
    """
    H, W = va.shape
    Af, Bf = _fill_invalid(A, B, va, vb)
    ga = cv2.cvtColor(Af, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(Bf, cv2.COLOR_BGR2GRAY)
    scale = 2 if W > 400 else 1
    if scale > 1:
        ga_s = cv2.resize(ga, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
        gb_s = cv2.resize(gb, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    else:
        ga_s, gb_s = ga, gb
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    fab = dis.calc(ga_s, gb_s, None)
    fba = dis.calc(gb_s, ga_s, None)
    if scale > 1:
        fab = cv2.resize(fab, (W, H), interpolation=cv2.INTER_LINEAR) * scale
        fba = cv2.resize(fba, (W, H), interpolation=cv2.INTER_LINEAR) * scale
    max_flow = max_flow_frac * W
    fab = np.clip(fab, -max_flow, max_flow)
    fba = np.clip(fba, -max_flow, max_flow)

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def fb_consistency(f_fwd, f_bwd):
        # f_fwd(p) + f_bwd(p + f_fwd(p)) should be ~0
        px = uu + f_fwd[..., 0]
        py = vv + f_fwd[..., 1]
        bx = cv2.remap(f_bwd[..., 0], px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        by = cv2.remap(f_bwd[..., 1], px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.hypot(f_fwd[..., 0] + bx, f_fwd[..., 1] + by)
        mag = np.hypot(f_fwd[..., 0], f_fwd[..., 1])
        return err < (fb_tol_px + 0.1 * mag)

    def clean(flow, ok):
        # zero unreliable flow, then smooth so the warp stays gentle
        flow = flow * ok[..., None].astype(np.float32)
        okf = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), 6.0)
        flow = cv2.GaussianBlur(flow, (0, 0), 6.0)
        # renormalise by the blurred validity so good flow isn't diluted
        flow = flow / np.maximum(okf, 0.15)[..., None] * np.minimum(okf * 2.0, 1.0)[..., None]
        return flow

    ok_ab = fb_consistency(fab, fba)
    ok_ba = fb_consistency(fba, fab)
    fab = clean(fab, ok_ab)
    fba = clean(fba, ok_ba)

    w = np.clip(1.0 - np.abs(uu - seam[:, None].astype(np.float32)) / float(taper_px), 0.0, 1.0)
    ov = (va & vb).astype(np.uint8)
    k = max(3, int(taper_px * 0.5) | 1)
    ov = cv2.dilate(ov, np.ones((k, k), np.uint8))
    ov = cv2.GaussianBlur(ov.astype(np.float32), (0, 0), taper_px * 0.15)
    w = w * np.clip(ov, 0.0, 1.0)

    def half_warp(img, flow):
        # fixed point iteration: p = q - 0.5 w flow(p), so the warped image at q
        # shows what img shows at p, half way to the other lens
        px, py = uu.copy(), vv.copy()
        for _ in range(3):
            fx = cv2.remap(flow[..., 0], px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.remap(flow[..., 1], px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            px = uu - 0.5 * w * fx
            py = vv - 0.5 * w * fy
        return cv2.remap(img, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    A2 = half_warp(Af, fab)
    B2 = half_warp(Bf, fba)
    mag = np.hypot(fab[..., 0], fab[..., 1]) * w
    return A2, B2, mag


# ----------------------------------------------------------------------------
# blending
# ----------------------------------------------------------------------------

def multiband_blend(A: np.ndarray, B: np.ndarray, maskA: np.ndarray, levels: int) -> np.ndarray:
    """Laplacian pyramid blend of A (where maskA==1) and B (where 0)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    M = maskA.astype(np.float32)
    h, w = M.shape
    levels = max(1, min(levels, int(math.log2(max(4, min(h, w)))) - 2))
    gA, gB, gM = [A], [B], [M]
    for _ in range(levels):
        gA.append(cv2.pyrDown(gA[-1]))
        gB.append(cv2.pyrDown(gB[-1]))
        gM.append(cv2.pyrDown(gM[-1]))
    out = None
    for i in range(levels, -1, -1):
        if i == levels:
            la, lb = gA[i], gB[i]
        else:
            size = (gA[i].shape[1], gA[i].shape[0])
            la = gA[i] - cv2.pyrUp(gA[i + 1], dstsize=size)
            lb = gB[i] - cv2.pyrUp(gB[i + 1], dstsize=size)
        m = gM[i][..., None]
        layer = la * m + lb * (1.0 - m)
        if out is None:
            out = layer
        else:
            out = cv2.pyrUp(out, dstsize=(layer.shape[1], layer.shape[0])) + layer
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# exposure matching
# ----------------------------------------------------------------------------

def match_gains(A: np.ndarray, B: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Per-channel gain g so that A*g^-0.5 and B*g^0.5 agree... returned as
    (gainA[3], gainB[3])."""
    a = cv2.GaussianBlur(A, (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(B, (0, 0), 3).astype(np.float32)
    ok = overlap & (a.max(axis=2) < 240) & (b.max(axis=2) < 240) & (a.min(axis=2) > 8) & (b.min(axis=2) > 8)
    if ok.sum() < 200:
        return np.ones(3, np.float32), np.ones(3, np.float32)
    ratio = np.median(a[ok] / b[ok], axis=0)
    ratio = np.clip(ratio, 0.5, 2.0)
    return (1.0 / np.sqrt(ratio)).astype(np.float32), np.sqrt(ratio).astype(np.float32)


def apply_gain(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    if np.allclose(gain, 1.0, atol=1e-3):
        return img
    lut = np.clip(np.arange(256, dtype=np.float32)[:, None] * gain[None, :], 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut.reshape(256, 1, 3))


# ----------------------------------------------------------------------------
# JPEG output with 360 metadata
# ----------------------------------------------------------------------------

def _jpeg_segments(data: bytes):
    """Yield (marker, segment_bytes) for the APPn segments following SOI."""
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF and 0xE0 <= data[i + 1] <= 0xEF:
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        yield data[i + 1], data[i:i + 2 + ln]
        i += 2 + ln


def gpano_xmp(width: int, height: int, heading: float = 0.0) -> bytes:
    xmp = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">'
        '<GPano:ProjectionType>equirectangular</GPano:ProjectionType>'
        '<GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>'
        f'<GPano:FullPanoWidthPixels>{width}</GPano:FullPanoWidthPixels>'
        f'<GPano:FullPanoHeightPixels>{height}</GPano:FullPanoHeightPixels>'
        f'<GPano:CroppedAreaImageWidthPixels>{width}</GPano:CroppedAreaImageWidthPixels>'
        f'<GPano:CroppedAreaImageHeightPixels>{height}</GPano:CroppedAreaImageHeightPixels>'
        '<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>'
        '<GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>'
        f'<GPano:PoseHeadingDegrees>{heading:.1f}</GPano:PoseHeadingDegrees>'
        '<GPano:StitchingSoftware>insta360stitch</GPano:StitchingSoftware>'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp.encode("utf-8")
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def save_pano_jpeg(path, pano: np.ndarray, quality: int = 95, source_jpeg: Optional[bytes] = None) -> None:
    ok, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    data = buf.tobytes()
    head = data[:2]
    rest = data[2:]
    extra = b""
    if source_jpeg:
        for marker, seg in _jpeg_segments(source_jpeg):
            if marker == 0xE1 and seg[4:10] == b"Exif\x00\x00":
                extra += seg
                break
    extra += gpano_xmp(pano.shape[1], pano.shape[0])
    # keep a JFIF APP0 first if the encoder wrote one
    if rest[:2] == b"\xff\xe0":
        ln = struct.unpack(">H", rest[2:4])[0]
        app0, rest = rest[:2 + ln], rest[2 + ln:]
        out = head + app0 + extra + rest
    else:
        out = head + extra + rest
    Path(path).write_bytes(out)


# ----------------------------------------------------------------------------
# whole-panorama helpers
# ----------------------------------------------------------------------------

def rotate_equirect(pano: np.ndarray, R: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Resample an equirectangular image so that output direction d shows the
    input at direction R @ d (used for levelling and yaw offsets)."""
    H, W = pano.shape[:2]
    out = np.empty_like(pano)
    RT = R.T.astype(np.float32)
    for y0 in range(0, H, chunk):
        y1 = min(H, y0 + chunk)
        d = equirect_dirs(W, H, y0, y1) @ RT
        u, v = dirs_to_equirect(d, W, H)
        out[y0:y1] = cv2.remap(pano, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_WRAP)
    return out


def feathered_fallback(primary: np.ndarray, valid: np.ndarray, other: np.ndarray, feather_px: float) -> np.ndarray:
    """primary where it is valid, fading into `other` across `feather_px`."""
    if valid.all():
        return primary
    dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip(dist / max(feather_px, 1.0), 0.0, 1.0)[..., None].astype(np.float32)
    return (primary.astype(np.float32) * w + other.astype(np.float32) * (1.0 - w)).astype(np.uint8)


# ----------------------------------------------------------------------------
# the stitcher
# ----------------------------------------------------------------------------

class Stitcher:
    """Stitches .insp files.  Keeps a per-camera-serial cache of refined lens
    models so later photos from the same camera start from a good solution."""

    def __init__(self, calib_cache: Optional[Dict[str, dict]] = None):
        self.calib_cache: Dict[str, dict] = calib_cache if calib_cache is not None else {}

    # -- lens model ---------------------------------------------------------
    def initial_lenses(self, insp: InspFile, img: np.ndarray, opts: StitchOptions) -> List[Lens]:
        if insp.calibration is None:
            raise ValueError(f"{insp.path.name}: no lens calibration found in file")
        lenses = lenses_from_calibration(insp.calibration, img.shape[1], img.shape[0], fov=opts.default_fov)
        cached = self.calib_cache.get(insp.serial or "unknown")
        if cached:
            for l in lenses:
                l.fov = float(cached.get("fov", l.fov))
                l.k1 = float(cached.get("k1", l.k1))
                l.k2 = float(cached.get("k2", l.k2))
            lenses[1].yaw = float(cached.get("yaw2", lenses[1].yaw))
            lenses[1].pitch = float(cached.get("pitch2", lenses[1].pitch))
            lenses[1].roll = float(cached.get("roll2", lenses[1].roll))
        return lenses

    def remember(self, serial: str, lenses: List[Lens]):
        self.calib_cache[serial or "unknown"] = {
            "fov": lenses[0].fov, "k1": lenses[0].k1, "k2": lenses[0].k2,
            "yaw2": lenses[1].yaw, "pitch2": lenses[1].pitch, "roll2": lenses[1].roll,
        }

    # -- main entry ---------------------------------------------------------
    def stitch(self, insp: InspFile, opts: StitchOptions = StitchOptions(), progress: Progress = None):
        t0 = time.time()
        log = progress or (lambda s: None)
        info: dict = {"file": str(insp.path), "options": opts.to_dict()}

        img = cv2.imdecode(np.frombuffer(insp.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{insp.path.name}: could not decode JPEG")
        lenses = self.initial_lenses(insp, img, opts)

        if opts.refine:
            log("aligning lenses (control points)")
            res = calibrate(img, lenses, log=log)
            info["alignment"] = res.message
            info["alignment_points"] = res.n_points
            log(res.message)
            if res.accepted:
                lenses = res.lenses
                self.remember(insp.serial, lenses)
        info["lenses"] = [l.to_dict() for l in lenses]

        # Stitch in the camera frame (seams exactly at +-90 deg), level at the end.
        R = np.eye(3)
        accel = insp.accel
        if opts.level and accel is not None:
            up = up_from_accel(accel)
            R = level_rotation(up)
            info["levelled"] = True
            info["tilt_deg"] = float(math.degrees(math.acos(np.clip(up[1], -1, 1))))
        else:
            info["levelled"] = False
        if opts.yaw_offset:
            R = R @ rot_y(opts.yaw_offset)

        W = int(opts.width)
        H = W // 2
        I3 = np.eye(3)
        log(f"projecting lenses to {W}x{H}")
        img1, v1 = render_lens(img, lenses[0], I3, W, H)
        img2, v2 = render_lens(img, lenses[1], I3, W, H)

        bands = []
        for lon_c in (-90.0, 90.0):
            u0 = int(round(lon_to_u(lon_c - opts.band_half_deg, W)))
            u1 = int(round(lon_to_u(lon_c + opts.band_half_deg, W)))
            bands.append((lon_c, u0, u1))

        ov_all = np.zeros((H, W), bool)
        for _, u0, u1 in bands:
            ov_all[:, u0:u1] = v1[:, u0:u1] & v2[:, u0:u1]
        gA, gB = match_gains(img1, img2, ov_all)
        info["gains"] = {"front": gA.tolist(), "back": gB.tolist()}
        img1 = apply_gain(img1, gA)
        img2 = apply_gain(img2, gB)

        feather = 0.01 * W
        out = feathered_fallback(img2, v2, img1, feather)
        (_, l0, l1), (_, r0, r1) = bands
        out[:, l1:r0] = feathered_fallback(img1[:, l1:r0], v1[:, l1:r0], img2[:, l1:r0], feather)

        info["seams"] = []
        for lon_c, u0, u1 in bands:
            log(f"seam at {lon_c:+.0f} deg: finding seam")
            A = img1[:, u0:u1]
            B = img2[:, u0:u1]
            va = v1[:, u0:u1]
            vb = v2[:, u0:u1]
            overlap = va & vb
            # guarantee a path even in rows without overlap (shouldn't happen)
            empty_rows = ~overlap.any(axis=1)
            if empty_rows.any():
                overlap[empty_rows, (u1 - u0) // 2] = True
            seam = min_cost_seam(seam_cost(A, B, overlap))
            flow_mag = None
            if opts.parallax:
                log(f"seam at {lon_c:+.0f} deg: optical flow parallax compensation")
                taper = 0.7 * (u1 - u0) / 2.0
                A, B, flow_mag = flow_warp(A, B, va, vb, seam, taper)
                seam = min_cost_seam(seam_cost(A, B, overlap))
            uu = np.arange(u1 - u0)[None, :]
            if lon_c < 0:
                maskA = (uu > seam[:, None])
            else:
                maskA = (uu < seam[:, None])
            log(f"seam at {lon_c:+.0f} deg: blending")
            out[:, u0:u1] = multiband_blend(A, B, maskA, opts.blend_levels)
            entry = {"lon": lon_c, "seam_mean_lon": float(((u0 + seam.mean()) / W * 2 - 1) * 180)}
            if flow_mag is not None:
                entry["flow_px_mean"] = float(flow_mag[overlap].mean())
                entry["flow_px_max"] = float(flow_mag[overlap].max())
            info["seams"].append(entry)

        if not np.allclose(R, I3):
            log("levelling the panorama")
            out = rotate_equirect(out, R)

        info["seconds"] = round(time.time() - t0, 2)
        log(f"done in {info['seconds']} s")
        return out, info


def stitch_file(src, dst=None, opts: StitchOptions = StitchOptions(), stitcher: Optional[Stitcher] = None,
                progress: Progress = None) -> Tuple[Path, dict]:
    """Stitch one .insp file to a JPEG (and a .json sidecar). Returns (path, info)."""
    from .insp import borrow_imu, find_insp_files, parse_insp

    src = Path(src)
    if dst is None:
        dst = src.with_suffix("").with_name(src.stem + "_pano.jpg")
    dst = Path(dst)
    stitcher = stitcher or Stitcher()
    insp = parse_insp(src)
    if opts.level and borrow_imu(insp, find_insp_files(src.parent)) and progress:
        progress("using IMU data from a sibling shot")
    pano, info = stitcher.stitch(insp, opts, progress)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_pano_jpeg(dst, pano, opts.jpeg_quality, insp.jpeg)
    dst.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return dst, info

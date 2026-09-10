"""Self-contained tests: a synthetic panorama is rendered through the lens
model into a fake .insp file (JPEG + Insta360 trailer), parsed back and
stitched.  Run with:  python -m unittest discover -s tests -v"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insta360stitch.geometry import Lens, dirs_to_equirect, equirect_dirs, level_rotation, rot_y  # noqa: E402
from insta360stitch.insp import TRAILER_MAGIC, Calibration, parse_insp  # noqa: E402
from insta360stitch.stitch import Stitcher, StitchOptions, min_cost_seam, multiband_blend, up_from_accel  # noqa: E402

OFFSET = "2_1480.043_1529.441_1532.750_0.000_0.000_0.000_1480.996_4544.292_1523.460_-0.044_0.049_-178.644_6080_3040_2323"


def synthetic_pano(w=2048, h=1024, seed=1) -> np.ndarray:
    """A colourful, feature-rich equirectangular test scene."""
    rng = np.random.default_rng(seed)
    pano = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    pano[..., 0] = (xx * 255 // w)
    pano[..., 1] = (yy * 255 // h)
    pano[..., 2] = ((xx // 64 + yy // 64) % 2) * 120 + 60
    for _ in range(400):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, 40))
        cv2.circle(pano, (x, y), r, c, -1)
    return cv2.GaussianBlur(pano, (0, 0), 1.0)


def _pb_string(field: int, s: bytes) -> bytes:
    return bytes([(field << 3) | 2, len(s)]) + s


def make_fake_insp(path: Path, pano: np.ndarray, calib_str=OFFSET, img_w=6080, img_h=3040, scale=0.25,
                   with_imu=True, lens_fov=200.0, yaw2=0.0):
    """Render two fisheye images from `pano` and wrap them in an .insp."""
    cal = Calibration.from_string(calib_str)
    W, H = int(img_w * scale), int(img_h * scale)
    centres = cal.stored_centres(img_w, img_h)
    lenses = []
    for i, (lc, (X, Y, r)) in enumerate(zip(cal.lenses, centres)):
        lenses.append(Lens(cx=X * scale, cy=Y * scale, r_cal=r * scale, fov=lens_fov, yaw=lc.yaw + (yaw2 if i else 0),
                           pitch=lc.pitch, roll=lc.roll, flip=(i == 1)))
    img = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for lens in lenses:
        v = lens.unproject(xx, yy)
        inside = np.hypot(xx - lens.cx, yy - lens.cy) < lens.r_cal
        d = v @ lens.M()  # lens -> world
        u, vv = dirs_to_equirect(d, pano.shape[1], pano.shape[0])
        samp = cv2.remap(pano, u.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        img[inside] = samp[inside]
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    jpeg = jpeg.tobytes()
    # scaled calibration string (same layout as the real one)
    parts = calib_str.split("_")
    nums = [float(x) for x in parts[1:]]
    for i in (0, 1, 2, 6, 7, 8):
        nums[i] *= scale
    nums[12], nums[13] = W, H
    cs = parts[0] + "_" + "_".join(f"{n:.3f}" if i not in (12, 13, 14) else f"{int(n)}" for i, n in enumerate(nums))
    info = _pb_string(1, b"TESTSERIAL0001") + _pb_string(2, b"Insta360 One2") + _pb_string(3, b"v9.9.9_build1") + _pb_string(5, cs.encode())
    records = [(0x0101, info)]
    if with_imu:
        # camera upright: gravity on sensor -x  ->  accel (-1, 0, 0)
        samples = b"".join(struct.pack("<Q6d", t, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0) for t in range(20))
        records.append((0x0300, samples))
    trailer = b""
    for rid, data in records:
        trailer += data + struct.pack("<HI", rid, len(data))
    trailer += b"\x00" * 32
    trailer_len = len(trailer) + 8 + 32
    trailer += struct.pack("<II", trailer_len, 3) + TRAILER_MAGIC
    path.write_bytes(jpeg + trailer)
    return lenses


class GeometryTests(unittest.TestCase):
    def test_lens_round_trip(self):
        L = Lens(cx=400, cy=380, r_cal=383, fov=200, k1=0.01, k2=-0.002, yaw=1, pitch=-2, roll=-178.6, flip=True)
        d = equirect_dirs(128, 64)
        v = d @ L.M().T
        X, Y, th = L.project(v)
        v2 = L.unproject(X, Y)
        ok = th < L.theta_max
        self.assertLess(np.abs(v2 - v)[ok].max(), 1e-5)

    def test_level_rotation(self):
        up = up_from_accel(np.array([-0.98, -0.2, 0.0]))
        up = up / np.linalg.norm(up)
        R = level_rotation(up)
        self.assertTrue(np.allclose(R @ np.array([0, 1.0, 0]), up, atol=1e-9))
        self.assertTrue(np.allclose(level_rotation(np.array([0, 1.0, 0])), np.eye(3)))

    def test_seam_and_blend(self):
        cost = np.ones((50, 30), np.float32) * 10
        cost[:, 12] = 0.1
        seam = min_cost_seam(cost)
        self.assertTrue(np.all(seam == 12))
        A = np.full((64, 64, 3), 200, np.uint8)
        B = np.full((64, 64, 3), 50, np.uint8)
        mask = np.zeros((64, 64), bool)
        mask[:, :32] = True
        out = multiband_blend(A, B, mask, 4)
        self.assertGreater(int(out[:, :8].mean()), 190)
        self.assertLess(int(out[:, -8:].mean()), 60)


class InspTests(unittest.TestCase):
    def test_parse_real_layout(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.insp"
            make_fake_insp(p, synthetic_pano(512, 256), scale=0.1)
            f = parse_insp(p)
            self.assertEqual(f.serial, "TESTSERIAL0001")
            self.assertEqual(f.model, "Insta360 One2")
            self.assertIsNotNone(f.calibration)
            self.assertEqual(len(f.calibration.lenses), 2)
            self.assertIsNotNone(f.imu)
            self.assertEqual(f.imu.shape, (20, 7))
            self.assertTrue(np.allclose(f.accel, [-1, 0, 0]))
            self.assertTrue(f.jpeg.startswith(b"\xff\xd8") and f.jpeg.endswith(b"\xff\xd9"))


class StitchTests(unittest.TestCase):
    def _run(self, **kw):
        pano = synthetic_pano()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "IMG_test.insp"
            make_fake_insp(p, pano, **kw.pop("insp", {}))
            f = parse_insp(p)
            opts = StitchOptions(width=1024, **kw)
            out, info = Stitcher().stitch(f, opts)
        ref = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
        return out, ref, info

    def test_stitch_matches_source(self):
        out, ref, info = self._run(parallax=True, refine=False, level=False)
        err = np.abs(out.astype(np.float32) - ref.astype(np.float32)).mean()
        self.assertLess(err, 6.0, f"mean abs error {err}")
        self.assertEqual(out.shape, (512, 1024, 3))

    def test_refinement_recovers_yaw(self):
        # image rendered with a rear lens yawed by 3 degrees relative to the calibration string
        out, ref, info = self._run(parallax=False, refine=True, level=False, insp={"yaw2": 3.0})
        yaw = info["lenses"][1]["yaw"]
        self.assertAlmostEqual(yaw, -0.044 + 3.0, delta=0.5, msg=info.get("alignment"))
        err = np.abs(out.astype(np.float32) - ref.astype(np.float32)).mean()
        self.assertLess(err, 8.0)

    def test_levelling_keeps_upright_camera(self):
        out, ref, info = self._run(parallax=False, refine=False, level=True)
        self.assertTrue(info["levelled"])
        self.assertLess(info["tilt_deg"], 0.01)
        err = np.abs(out.astype(np.float32) - ref.astype(np.float32)).mean()
        self.assertLess(err, 6.0)


if __name__ == "__main__":
    unittest.main()

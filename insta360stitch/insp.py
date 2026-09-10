"""Reader for Insta360 .insp files.

An .insp file is an ordinary JPEG (the two fisheye circles side by side)
followed by a proprietary trailer.  The trailer is a list of records, each
terminated by a 6 byte footer (uint16 id, uint32 length), then 32 zero bytes,
a uint32 trailer length, a uint32 version and a 32 byte magic string.

Records we understand:
  0x0101  protobuf with serial number, model, firmware and the lens
          calibration ("offset") string
  0x0200  thumbnail JPEG
  0x0300  IMU samples: uint64 timestamp + 3 doubles accel + 3 doubles gyro
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TRAILER_MAGIC = b"8db42d694ccc418790edff439fe026bf"
OFFSET_RE = re.compile(rb"(\d+)_(-?[\d.]+(?:_-?[\d.]+){13,})")


@dataclass
class LensCal:
    """One lens as described by the factory offset string.

    cx, cy and r are in the calibration frame (the sensor's native portrait
    orientation, see `Calibration.stored_centres`).  yaw/pitch/roll are in
    degrees.
    """

    cx: float
    cy: float
    r: float
    yaw: float
    pitch: float
    roll: float


@dataclass
class Calibration:
    version: int
    lenses: List[LensCal]
    width: int
    height: int
    extra: List[float] = field(default_factory=list)
    raw: str = ""

    @classmethod
    def from_string(cls, s: str) -> "Calibration":
        parts = s.split("_")
        version = int(parts[0])
        nums = [float(x) for x in parts[1:]]
        lenses = [LensCal(*nums[0:6]), LensCal(*nums[6:12])]
        width, height = int(nums[12]), int(nums[13])
        return cls(version, lenses, width, height, nums[14:], s)

    def stored_centres(self, img_w: int, img_h: int):
        """Return [(X, Y, r), (X, Y, r)] lens centres in the stored JPEG frame.

        The ONE X writes the calibration in the sensor's portrait frame
        (3040 x 6080) while the JPEG is stored landscape (6080 x 3040), so the
        axes are swapped.  Detect that from the numbers instead of assuming.
        """
        max_cy = max(l.cy for l in self.lenses)
        max_cx = max(l.cx for l in self.lenses)
        if max_cy > img_h or (max_cx <= img_h and max_cy > img_w / 2):
            return [(l.cy, l.cx, l.r) for l in self.lenses]
        return [(l.cx, l.cy, l.r) for l in self.lenses]


@dataclass
class InspFile:
    path: Path
    jpeg: bytes
    records: Dict[int, bytes]
    serial: str = ""
    model: str = ""
    firmware: str = ""
    calibration: Optional[Calibration] = None
    imu: Optional[np.ndarray] = None  # (N, 7): t, ax, ay, az, gx, gy, gz

    @property
    def thumbnail(self) -> Optional[bytes]:
        return self.records.get(0x0200)

    @property
    def accel(self) -> Optional[np.ndarray]:
        """Mean accelerometer vector (gravity, in g) or None."""
        if self.imu is None or len(self.imu) == 0:
            return None
        a = self.imu[:, 1:4]
        # take the quietest half of the samples so hand shake is ignored
        norms = np.linalg.norm(a, axis=1)
        keep = np.abs(norms - np.median(norms)) <= np.percentile(np.abs(norms - np.median(norms)), 50)
        v = a[keep].mean(axis=0) if keep.any() else a.mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None


def _protobuf_fields(buf: bytes):
    """Very small protobuf walker: yields (field_no, wire_type, value)."""
    i = 0
    n = len(buf)

    def varint(i):
        shift = 0
        val = 0
        while i < n:
            b = buf[i]
            i += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        return val, i

    while i < n:
        try:
            key, i = varint(i)
        except Exception:
            return
        fno, wt = key >> 3, key & 7
        if wt == 0:
            val, i = varint(i)
        elif wt == 1:
            val = buf[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = varint(i)
            val = buf[i:i + ln]
            i += ln
        elif wt == 5:
            val = buf[i:i + 4]
            i += 4
        else:
            return
        yield fno, wt, val


def _parse_imu(rec: bytes) -> Optional[np.ndarray]:
    for fmt, size in (("<Q6d", 56), ("<Q6f", 32)):
        if len(rec) % size == 0 and len(rec) >= size:
            arr = np.array([struct.unpack(fmt, rec[i:i + size]) for i in range(0, len(rec), size)], dtype=np.float64)
            norm = np.linalg.norm(arr[:, 1:4], axis=1)
            if 0.5 < np.median(norm) < 2.0:
                return arr
    return None


def parse_insp(path) -> InspFile:
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"{path.name}: not a JPEG based .insp file")
    records: Dict[int, bytes] = {}
    jpeg = data
    if data.endswith(TRAILER_MAGIC) and len(data) > 72:
        trailer_len, _version = struct.unpack("<II", data[-40:-32])
        start = len(data) - trailer_len
        if 0 < start < len(data):
            jpeg = data[:start]
            pos = len(data) - 72  # skip magic, len/version and 32 zero bytes
            while pos - 6 >= start:
                rid, rlen = struct.unpack("<HI", data[pos - 6:pos])
                if rlen > pos - 6 - start + 0:
                    break
                records[rid] = data[pos - 6 - rlen:pos - 6]
                pos = pos - 6 - rlen
    f = InspFile(path=path, jpeg=jpeg, records=records)

    info = records.get(0x0101)
    if info:
        for fno, wt, val in _protobuf_fields(info):
            if wt != 2 or not isinstance(val, (bytes, bytearray)):
                continue
            text = bytes(val)
            m = OFFSET_RE.search(text)
            if m and f.calibration is None:
                try:
                    f.calibration = Calibration.from_string(m.group(0).decode())
                except Exception:
                    pass
            elif b"Insta360" in text and not f.model:
                f.model = text.decode(errors="replace")
            elif re.fullmatch(rb"[A-Z0-9]{10,20}", text) and not f.serial:
                f.serial = text.decode()
            elif text.startswith(b"v") and b"build" in text and not f.firmware:
                f.firmware = text.decode(errors="replace")
    if f.calibration is None:
        m = OFFSET_RE.search(data[len(jpeg):] if len(jpeg) < len(data) else data[-4096:])
        if m:
            f.calibration = Calibration.from_string(m.group(0).decode())
    imu = records.get(0x0300)
    if imu:
        f.imu = _parse_imu(imu)
    return f


def find_insp_files(directory) -> List[Path]:
    directory = Path(directory)
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".insp"]
    return sorted(files)


def borrow_imu(insp: InspFile, candidates) -> bool:
    """Bracketed / burst shots share one exposure moment but only one of them
    carries the IMU record.  Copy it from a sibling with the same
    IMG_<date>_<time> prefix.  Returns True if something was borrowed."""
    if insp.imu is not None:
        return False
    stem = insp.path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return False
    prefix = "_".join(parts[:3])
    for other in candidates:
        other = Path(other)
        if other == insp.path or not other.stem.startswith(prefix):
            continue
        try:
            sib = parse_insp(other)
        except Exception:
            continue
        if sib.imu is not None:
            insp.imu = sib.imu
            return True
    return False

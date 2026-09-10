"""Command line entry point.

    python -m insta360stitch                 # open the GUI
    python -m insta360stitch /path/to/dir    # open the GUI on that folder
    python -m insta360stitch stitch DIR      # batch stitch without a GUI
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="insta360stitch", description="Stitch and view Insta360 ONE X .insp photos")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", help="open the desktop app (default)")
    g.add_argument("folder", nargs="?")

    s = sub.add_parser("stitch", help="stitch every .insp in a folder (or single files) without a GUI")
    s.add_argument("paths", nargs="+", help="folder(s) or .insp file(s)")
    s.add_argument("-o", "--out", help="output folder (default: <folder>/stitched)")
    s.add_argument("-w", "--width", type=int, default=6080, help="output width in pixels (default 6080)")
    s.add_argument("--no-parallax", action="store_true", help="disable optical-flow parallax compensation")
    s.add_argument("--no-level", action="store_true", help="do not level the horizon from the IMU")
    s.add_argument("--no-refine", action="store_true", help="use the factory lens model without refinement")
    s.add_argument("--yaw", type=float, default=0.0, help="rotate the panorama by this many degrees")
    s.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality")
    s.add_argument("-f", "--force", action="store_true", help="re-stitch even if the output exists")

    i = sub.add_parser("info", help="print the metadata found in .insp files")
    i.add_argument("paths", nargs="+")
    return p


def cmd_stitch(args) -> int:
    from .insp import find_insp_files
    from .stitch import Stitcher, StitchOptions, stitch_file

    opts = StitchOptions(width=args.width, parallax=not args.no_parallax, level=not args.no_level,
                         refine=not args.no_refine, yaw_offset=args.yaw, jpeg_quality=args.quality)
    stitcher = Stitcher()
    files = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            files.extend((f, Path(args.out) if args.out else p / "stitched") for f in find_insp_files(p))
        elif p.suffix.lower() == ".insp":
            files.append((p, Path(args.out) if args.out else p.parent / "stitched"))
        else:
            print(f"skipping {p}: not a folder or .insp file", file=sys.stderr)
    if not files:
        print("no .insp files found", file=sys.stderr)
        return 1
    failures = 0
    for n, (src, out_dir) in enumerate(files, 1):
        dst = out_dir / (src.stem + ".jpg")
        if dst.exists() and not args.force:
            print(f"[{n}/{len(files)}] {src.name}: exists, skipping (use --force)")
            continue
        print(f"[{n}/{len(files)}] {src.name}")
        try:
            path, info = stitch_file(src, dst, opts, stitcher, progress=lambda s: print(f"    {s}"))
            print(f"    -> {path}")
        except Exception as e:
            failures += 1
            print(f"    failed: {e}", file=sys.stderr)
    if stitcher.calib_cache:
        for out_dir in {o for _, o in files}:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "calibration.json").write_text(json.dumps(stitcher.calib_cache, indent=2))
            except Exception:
                pass
    return 1 if failures else 0


def cmd_info(args) -> int:
    from .insp import find_insp_files, parse_insp

    paths = []
    for p in args.paths:
        p = Path(p)
        paths.extend(find_insp_files(p) if p.is_dir() else [p])
    for p in paths:
        f = parse_insp(p)
        print(f"{p.name}: {f.model} serial={f.serial} firmware={f.firmware} jpeg={len(f.jpeg)} bytes")
        print(f"    records: {', '.join('0x%04x (%d bytes)' % (k, len(v)) for k, v in sorted(f.records.items()))}")
        if f.calibration:
            print(f"    calibration: {f.calibration.raw}")
        if f.imu is not None:
            print(f"    IMU: {len(f.imu)} samples, gravity {np_fmt(f.accel)}")
    return 0


def np_fmt(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")" if v is not None else "n/a"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # bare folder argument -> GUI on that folder
    if argv and argv[0] not in ("gui", "stitch", "info", "-h", "--help", "--version") and Path(argv[0]).is_dir():
        argv = ["gui"] + argv
    args = build_parser().parse_args(argv)
    if args.cmd == "stitch":
        return cmd_stitch(args)
    if args.cmd == "info":
        return cmd_info(args)
    from .app import main as gui_main

    gui_main(getattr(args, "folder", None))
    return 0

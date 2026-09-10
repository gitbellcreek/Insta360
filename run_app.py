#!/usr/bin/env python3
"""Double-click / run this to open the app.  Optional argument: a folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from insta360stitch.cli import main

sys.exit(main())

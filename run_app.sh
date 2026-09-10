#!/usr/bin/env bash
# Linux launcher: creates a virtual environment on first run, then starts the app.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi
if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    echo "Tkinter is missing. Install it with e.g.:  sudo apt install python3-tk   (Debian/Ubuntu)"
    echo "                                        sudo dnf install python3-tkinter (Fedora)"
    exit 1
fi
exec .venv/bin/python run_app.py "$@"

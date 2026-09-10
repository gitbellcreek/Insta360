@echo off
rem Windows launcher: creates a virtual environment on first run, then starts the app.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || (echo Python 3.9+ is required. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)
".venv\Scripts\python.exe" run_app.py %*
if errorlevel 1 pause
endlocal

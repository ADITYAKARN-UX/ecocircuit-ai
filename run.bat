@echo off
title EcoCircuit AI — Launcher
color 0A

echo.
echo  =========================================================
echo    EcoCircuit AI - Environmental Intelligence Platform
echo    Predict Pollution. Optimize Production. Reduce E-Waste.
echo  =========================================================
echo.

:: ── Step 1: Go to the project directory ──────────────────────────────────────
cd /d "%~dp0"

:: ── Step 2: Check Python is installed ────────────────────────────────────────
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  and make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
python --version
echo       Python found. OK
echo.

:: ── Step 3: Create virtual environment if it doesn't exist ───────────────────
echo [2/4] Setting up virtual environment...
if not exist "venv\" (
    echo       Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created. OK
) else (
    echo       Virtual environment already exists. Skipping.
)
echo.

:: ── Step 4: Activate venv and install dependencies ───────────────────────────
echo [3/4] Installing / verifying dependencies...
call venv\Scripts\activate.bat

pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Failed to install dependencies.
    echo  Try running: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo       All dependencies satisfied. OK
echo.

:: ── Step 5: Launch Streamlit app ─────────────────────────────────────────────
echo [4/4] Launching EcoCircuit AI...
echo.
echo  ----------------------------------------------------------
echo   App will open at: http://localhost:8501
echo   Press Ctrl+C in this window to stop the server.
echo  ----------------------------------------------------------
echo.

streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

:: ── On exit ──────────────────────────────────────────────────────────────────
echo.
echo  EcoCircuit AI has stopped.
pause

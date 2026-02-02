@echo off
echo Starting Heat Exchanger Dashboard...
cd /d "%~dp0"

:: Check if .venv exists
if not exist ".venv" (
    echo Virtual environment not found. Creating...
    py -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo Launching Streamlit...
streamlit run dashboard.py

pause

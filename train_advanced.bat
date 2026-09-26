@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" train_advanced.py
) else if exist ".venv\bin\python.exe" (
    ".venv\bin\python.exe" train_advanced.py
) else (
    echo Project environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Advanced training finished. Results are in model_output_advanced.
pause

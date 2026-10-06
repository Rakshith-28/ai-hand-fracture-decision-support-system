@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" train.py
) else if exist ".venv\bin\python.exe" (
    ".venv\bin\python.exe" train.py
) else (
    echo Project environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Training finished. Results are in model_output.
pause

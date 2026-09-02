@echo off
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" set "VENV_PYTHON=.venv\bin\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo Training the balanced hand-fracture model...
"%VENV_PYTHON%" train_balanced.py
echo.
echo Training finished. Results are in model_output_balanced.
pause

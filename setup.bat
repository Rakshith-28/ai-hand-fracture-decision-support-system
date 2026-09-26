@echo off
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_COMMAND=py"
) else (
    set "PYTHON_COMMAND=python"
)

echo Creating the project environment...
%PYTHON_COMMAND% -m venv .venv
if errorlevel 1 goto error

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" set "VENV_PYTHON=.venv\bin\python.exe"
if not exist "%VENV_PYTHON%" goto error

echo Installing required packages...
"%VENV_PYTHON%" -m pip install -r requirements.txt
"%VENV_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto error

echo.
echo Setup complete. You can now run train_advanced.bat.
pause
exit /b 0

:error
echo.
echo Setup failed. Install Python 3.11 or newer and try again.
pause
exit /b 1

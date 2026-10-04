@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" test_unseen.py %*
) else if exist ".venv\bin\python.exe" (
    ".venv\bin\python.exe" test_unseen.py %*
) else (
    echo Project environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo.
pause

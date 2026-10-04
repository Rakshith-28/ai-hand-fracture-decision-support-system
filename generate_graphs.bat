@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" generate_graphs.py %*
) else if exist ".venv\bin\python.exe" (
    ".venv\bin\python.exe" generate_graphs.py %*
) else (
    echo Project environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Graph generation finished. Check the matching graph_outputs folder.
pause

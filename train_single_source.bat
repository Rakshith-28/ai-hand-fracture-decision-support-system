@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Building or resuming the single-source BoneFract dataset...
".venv\Scripts\python.exe" build_bonefract_single_source.py
if errorlevel 1 goto error

echo Training the anatomy-aware MobileNetV3-Large model...
".venv\Scripts\python.exe" train_single_source_multitask.py
if errorlevel 1 goto error

echo.
echo Complete. Results are in model_output_single_source.
pause
exit /b 0

:error
echo.
echo The pipeline stopped because an error occurred.
pause
exit /b 1

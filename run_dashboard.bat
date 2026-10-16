@echo off
rem Launch the Meridian Doctor/Therapist Web Dashboard
setlocal
cd /d "%~dp0"
set "PY=C:\Users\ASHUTOSH\AppData\Roaming\mamba\envs\.conda-env\python.exe"
if not exist "%PY%" (
  echo [ERROR] Conda environment not found at %PY%
  pause
  exit /b 1
)
echo Starting Meridian Web Dashboard at http://127.0.0.1:5000 ...
"%PY%" src\dashboard\app.py
endlocal

@echo off
rem Launch the interactive 3D robot digital twin simulation
setlocal
cd /d "%~dp0"
set "PY=C:\Users\ASHUTOSH\AppData\Roaming\mamba\envs\.conda-env\python.exe"
if not exist "%PY%" (
  echo [ERROR] Conda 3D environment not found at %PY%
  pause
  exit /b 1
)
echo Starting 3D Robot Digital Twin...
"%PY%" -m robot_sim --manual %*
endlocal

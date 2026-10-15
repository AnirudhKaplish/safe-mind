@echo off
rem Start the Meridian dashboard and the robot digital twin; stop both on exit.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if "%PORT%"=="" set "PORT=5000"
set "URL=http://127.0.0.1:%PORT%/api/robot/state"

if not exist "results\bert_model" (
  echo ERROR: results\bert_model\ is missing ^(BERT weights^).
  echo You can still run: %PY% -m robot_sim --manual
  exit /b 1
)
dir /s /b results\*.joblib >nul 2>&1 || (
  echo ERROR: no baseline *.joblib artefacts under results\.
  exit /b 1
)

set "PYTHONPATH=."
start "mhd-dashboard" /B "%PY%" src\dashboard\app.py > runs_dashboard.log 2>&1

echo Waiting for the dashboard at %URL% ...
for /l %%i in (1,1,120) do (
  "%PY%" -c "import sys,urllib.request;urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1],timeout=0.5)" "%URL%" >nul 2>&1 && goto ready
  timeout /t 1 /nobreak >nul
)
echo ERROR: the dashboard did not answer. See runs_dashboard.log
goto stop

:ready
echo Dashboard: http://127.0.0.1:%PORT%
"%PY%" -m robot_sim --url "%URL%" %*

:stop
rem stop the dashboard (and its debug-reloader child)
for /f "tokens=2 delims=," %%p in ('wmic process where "commandline like '%%src\\dashboard\\app.py%%' and name like 'python%%'" get processid /format:csv ^| findstr /r "[0-9]"') do taskkill /PID %%p /T /F >nul 2>&1
endlocal

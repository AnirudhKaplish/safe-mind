@echo off
REM SAFE-MIND user app launcher (Windows)
cd /d D:\mhd-robot-integration
call venv\Scripts\activate

echo [1/2] Starting the consent blockchain bridge on port 8546...
start "SAFE-MIND blockchain" cmd /k "cd /d D:\mhd-robot-integration\blockchain && node scripts\consent_server.js"
timeout /t 6 >nul

echo [2/2] Starting dashboard + robot...
call run_robot_demo.bat

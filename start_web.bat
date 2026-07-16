@echo off
cd /d "%~dp0"
echo Starting NeuroScan (API + Frontend)...
powershell -ExecutionPolicy Bypass -File "%~dp0start_web.ps1"

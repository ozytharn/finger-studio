@echo off
cd /d "%~dp0"
if exist ".venv\pyvenv.cfg" (
    findstr /i "python3.14t" ".venv\pyvenv.cfg" >nul
    if not errorlevel 1 rmdir /s /q ".venv"
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.14 -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -c "import cv2, mediapipe, PIL" >nul 2>&1
if errorlevel 1 (
    echo Dependencies could not be imported. Delete .venv and run this file again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py

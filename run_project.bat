@echo off
title Virtual Health Assistant - XAI
cd /d "%~dp0"

echo ================================================
echo   VIRTUAL HEALTH ASSISTANT WITH XAI
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment already exists.
)

echo.
echo [2/5] Checking required packages...
venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Package installation failed.
    pause
    exit /b 1
)

echo.
echo [3/5] Training / updating models...
venv\Scripts\python.exe train_model.py

if errorlevel 1 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)

echo.
echo [4/5] Starting Flask backend...
start "Virtual Health Assistant - Flask Backend" cmd /k "cd /d ""%~dp0"" && venv\Scripts\python.exe app.py"

echo.
echo Waiting for Flask backend to start...
timeout /t 5 /nobreak >nul

echo.
echo [5/5] Starting Streamlit frontend...
start "Virtual Health Assistant - Streamlit" cmd /k "cd /d ""%~dp0"" && venv\Scripts\python.exe -m streamlit run streamlit_app.py"

echo.
echo ================================================
echo   PROJECT STARTED
echo ================================================
echo.
echo   Backend:  http://127.0.0.1:5000
echo   Frontend: http://localhost:8501
echo.
echo   Your browser should open automatically.
echo   Keep the Flask and Streamlit windows open.
echo ================================================
echo.

timeout /t 5 /nobreak >nul
start "" "http://localhost:8501"

exit /b 0
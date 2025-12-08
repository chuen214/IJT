@echo off
REM Injection Time Predictor - UI Launcher (Python venv version)
REM 适用于标准 Python 安装（不使用 Anaconda）

echo ========================================
echo  Injection Time Predictor
echo  Smart UI with Range Warning
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create virtual environment first:
    echo   python -m venv venv
    echo.
    pause
    exit /b 1
)

echo Starting UI...
echo.

REM Activate virtual environment and run streamlit
call venv\Scripts\activate.bat && streamlit run streamlit_app_improved.py

echo.
echo UI has been closed.
echo.
pause


@echo off
REM Setup Script for Python venv
REM 创建虚拟环境并安装依赖的一键脚本

echo ========================================
echo  Injection Time Predictor
echo  Setup Script (Python venv)
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8-3.11 first.
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists.
    echo.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
)

echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

echo [4/4] Installing dependencies...
echo This may take 5-10 minutes...
echo.
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages failed to install.
    echo Try using Chinese mirror:
    echo   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)
echo.

echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run: python check_environment.py
echo   2. Start UI: start_ui_venv.bat
echo.
pause


@echo off
echo.
echo ===============================================
echo    ROBOTIC ARM CONTROLLER LAUNCHER
echo ===============================================
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the 'venv' folder exists and contains the virtual environment.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo ✅ Virtual environment activated successfully!
echo.

:: Run the controller
echo 🚀 Starting Robotic Arm Controller...
echo.
python main.py

:: Deactivate virtual environment
echo.
echo 🧹 Deactivating virtual environment...
deactivate

echo.
echo ✅ Program finished. Press any key to exit...
pause >nul 
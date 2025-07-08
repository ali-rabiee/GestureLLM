Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   ROBOTIC ARM CONTROLLER LAUNCHER" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please make sure the 'venv' folder exists and contains the virtual environment." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "⚡ Activating virtual environment..." -ForegroundColor Yellow
try {
    & "venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated successfully!" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host "ERROR: Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the controller
Write-Host "🚀 Starting Robotic Arm Controller..." -ForegroundColor Green
Write-Host ""

try {
    python main.py
}
catch {
    Write-Host "ERROR: Failed to run the controller!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Deactivate virtual environment
Write-Host ""
Write-Host "🧹 Deactivating virtual environment..." -ForegroundColor Yellow
deactivate

Write-Host ""
Write-Host "✅ Program finished. Press any key to exit..." -ForegroundColor Green
Read-Host 
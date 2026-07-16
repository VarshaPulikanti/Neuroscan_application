# Quick commands (run from project root)
param(
    [ValidateSet("install", "data", "train", "eval", "web", "api", "frontend", "build", "test")]
    [string]$Action = "train"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if (Test-Path "C:\Python313\python.exe") { "C:\Python313\python.exe" } else { "python" }

switch ($Action) {
    "install"  { & $Python -m pip install -r requirements.txt }
    "data"     { & $Python scripts/prepare_data.py @args }
    "train"    { & $Python train.py @args }
    "eval"     { & $Python evaluate.py @args }
    "api"      { & $Python -m uvicorn app.api:app --reload --port 8000 }
    "frontend" { Set-Location frontend; npm run dev }
    "build"    { Set-Location frontend; npm install; npm run build; Set-Location .. }
    "web"      {
        Write-Host "Starting NeuroScan full-stack app..."
        Write-Host "  API:      http://localhost:8000"
        Write-Host "  Frontend: http://localhost:5173"
        Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.api:app", "--reload", "--port", "8000" -WorkingDirectory $PSScriptRoot
        Set-Location frontend
        if (-not (Test-Path "node_modules")) { npm install }
        npm run dev
    }
    "test"     { & $Python -m pytest tests/ -q }
}

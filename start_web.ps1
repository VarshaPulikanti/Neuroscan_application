# Launch NeuroScan full-stack app (API + React frontend)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if (Test-Path "C:\Python313\python.exe") { "C:\Python313\python.exe" } else { "python" }

Write-Host "Starting API on http://localhost:8000 ..."
Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.api:app", "--reload", "--port", "8000" -WorkingDirectory $PSScriptRoot

Set-Location frontend
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..."
    npm install
}

Write-Host "Starting frontend on http://localhost:5173 ..."
npm run dev

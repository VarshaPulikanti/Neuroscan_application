# Quick commands (run from project root)
param(
    [ValidateSet("install", "data", "train", "eval", "demo", "api", "test")]
    [string]$Action = "train"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if (Test-Path "C:\Python313\python.exe") { "C:\Python313\python.exe" } else { "python" }

switch ($Action) {
    "install" { & $Python -m pip install -r requirements.txt }
    "data"    { & $Python scripts/prepare_data.py @args }
    "train"   { & $Python train.py @args }
    "eval"    { & $Python evaluate.py @args }
    "demo"    { streamlit run app/streamlit_app.py }
    "api"     { & $Python -m uvicorn app.api:app --reload --port 8000 }
    "test"    { & $Python -m pytest tests/ -q }
}

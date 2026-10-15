# EchoTwin Windows bootstrap (PowerShell 5.1+).
# Creates .venv, installs the package with voice+train+ui extras, and runs doctor.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
}

uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[voice,train,ui,lang,dev]"
.\.venv\Scripts\python scripts/check_environment.py
.\.venv\Scripts\python -m pytest -q
Write-Host "Done. Try: .\.venv\Scripts\echotwin.exe demo"

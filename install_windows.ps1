[CmdletBinding()]
param([switch]$RecreateVenv, [switch]$WithDevTools)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Invoke-Checked([scriptblock]$Command, [string]$Description) {
  Write-Host "`n==> $Description" -ForegroundColor Cyan
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}
function Require([string]$Name) { if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "$Name not found in PATH" } }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
Require "py"; Require "ffmpeg"; Require "ffprobe"
& py -3.12 -c "import sys" *> $null
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required. Install: winget install --id Python.Python.3.12 --exact" }
if ($RecreateVenv -and (Test-Path .venv)) { Remove-Item -Recurse -Force .venv }
if (-not (Test-Path .venv\Scripts\python.exe)) { Invoke-Checked { py -3.12 -m venv .venv } "Creating Python 3.12 environment" }
$python = Join-Path $root ".venv\Scripts\python.exe"
Invoke-Checked { & $python -m pip install --upgrade pip setuptools wheel } "Updating packaging tools"
Invoke-Checked { & $python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126 } "Installing CUDA PyTorch"
$extras = if ($WithDevTools) { ".[dev]" } else { "." }
Invoke-Checked { & $python -m pip install -e $extras } "Installing project"
Invoke-Checked { & $python -c "import torch, cv2, ultralytics; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" } "Smoke test"
Write-Host "`nInstallation completed." -ForegroundColor Green
Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
Write-Host 'Run: goalkeeper-highlights "C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4" --frame-stride 2 --max-candidates 10 --overwrite'

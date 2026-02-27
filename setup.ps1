# setup.ps1 — end-to-end project bootstrap for Windows PowerShell
# Usage:  .\setup.ps1
# Runs:   venv creation → install → data collection → preprocessing →
#         voice prep → parallel training → debate

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot

# ── 1. Virtual environment ─────────────────────────────────────────────────────

$VenvDir = Join-Path $Root ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "`n[setup] Creating virtual environment …" -ForegroundColor Cyan
    python -m venv $VenvDir
}

$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
Write-Host "[setup] Activating virtual environment …" -ForegroundColor Cyan
. $Activate

# ── 2. Dependencies ────────────────────────────────────────────────────────────

Write-Host "`n[setup] Installing PyTorch (CUDA 12.8) …" -ForegroundColor Cyan
pip install --quiet torch torchvision torchaudio `
    --extra-index-url https://download.pytorch.org/whl/cu128

Write-Host "[setup] Installing remaining requirements …" -ForegroundColor Cyan
pip install --quiet -r (Join-Path $Root "requirements.txt")

# ── 3. Data collection ─────────────────────────────────────────────────────────

Write-Host "`n[setup] Collecting raw data …" -ForegroundColor Cyan
python (Join-Path $Root "scripts\collect_data.py")

# ── 4. Preprocessing ───────────────────────────────────────────────────────────

Write-Host "`n[setup] Preprocessing data …" -ForegroundColor Cyan
python (Join-Path $Root "scripts\preprocess_data.py")

# ── 5. Voice reference prep ────────────────────────────────────────────────────

Write-Host "`n[setup] Preparing voice references …" -ForegroundColor Cyan
python (Join-Path $Root "scripts\prepare_voices.py")

# ── 6. Parallel training ───────────────────────────────────────────────────────

Write-Host "`n[setup] Starting Trump training in background …" -ForegroundColor Cyan
$TrumpJob = Start-Process python `
    -ArgumentList (Join-Path $Root "scripts\train_trump.py") `
    -PassThru -NoNewWindow

Write-Host "[setup] Starting Biden training in background …" -ForegroundColor Cyan
$BidenJob = Start-Process python `
    -ArgumentList (Join-Path $Root "scripts\train_biden.py") `
    -PassThru -NoNewWindow

Write-Host "[setup] Waiting for both training jobs to finish …" -ForegroundColor Yellow
$TrumpJob.WaitForExit()
$BidenJob.WaitForExit()

if ($TrumpJob.ExitCode -ne 0) { Write-Warning "Trump training exited with code $($TrumpJob.ExitCode)" }
if ($BidenJob.ExitCode -ne 0) { Write-Warning "Biden training exited with code $($BidenJob.ExitCode)" }

# ── 7. Launch debate ───────────────────────────────────────────────────────────

Write-Host "`n[setup] Setup complete — launching debate …`n" -ForegroundColor Green
python (Join-Path $Root "scripts\start_debate.py") --persona both --tts auto --turns 5

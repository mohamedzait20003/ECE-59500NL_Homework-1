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

$BidenRaw  = Join-Path $Root "data\raw\biden\biden_debate_2020_1.txt"
$TrumpRaw  = Join-Path $Root "data\raw\trump\trump_debate_2020_1.txt"
if ((Test-Path $BidenRaw) -and (Test-Path $TrumpRaw)) {
    Write-Host "`n[setup] Raw data already exists — skipping collection." -ForegroundColor Yellow
} else {
    Write-Host "`n[setup] Collecting raw data …" -ForegroundColor Cyan
    python (Join-Path $Root "scripts\collect_data.py")
}

# ── 4. Preprocessing ───────────────────────────────────────────────────────────

$BidenTrain = Join-Path $Root "data\processed\biden_train.txt"
$TrumpTrain = Join-Path $Root "data\processed\trump_train.txt"
if ((Test-Path $BidenTrain) -and (Test-Path $TrumpTrain)) {
    Write-Host "`n[setup] Processed data already exists — skipping preprocessing." -ForegroundColor Yellow
} else {
    Write-Host "`n[setup] Preprocessing data …" -ForegroundColor Cyan
    python (Join-Path $Root "scripts\preprocess_data.py")
}

# ── 5. Voice reference prep ────────────────────────────────────────────────────

$BidenVoice = Join-Path $Root "data\voices\biden_reference.wav"
$TrumpVoice = Join-Path $Root "data\voices\trump_reference.wav"
if ((Test-Path $BidenVoice) -and (Test-Path $TrumpVoice)) {
    Write-Host "`n[setup] Voice references already exist — skipping voice prep." -ForegroundColor Yellow
} else {
    Write-Host "`n[setup] Preparing voice references …" -ForegroundColor Cyan
    python (Join-Path $Root "scripts\prepare_voices.py")
}

# ── 6. Parallel training ───────────────────────────────────────────────────────

$BidenModel = Join-Path $Root "models\biden\model.safetensors"
$TrumpModel = Join-Path $Root "models\trump\model.safetensors"
$TrumpJob   = $null
$BidenJob   = $null

if (Test-Path $TrumpModel) {
    Write-Host "`n[setup] Trump model already exists — skipping training." -ForegroundColor Yellow
} else {
    Write-Host "`n[setup] Starting Trump training in background …" -ForegroundColor Cyan
    $TrumpJob = Start-Process python `
        -ArgumentList (Join-Path $Root "scripts\train_trump.py") `
        -PassThru -NoNewWindow
}

if (Test-Path $BidenModel) {
    Write-Host "[setup] Biden model already exists — skipping training." -ForegroundColor Yellow
} else {
    Write-Host "[setup] Starting Biden training in background …" -ForegroundColor Cyan
    $BidenJob = Start-Process python `
        -ArgumentList (Join-Path $Root "scripts\train_biden.py") `
        -PassThru -NoNewWindow
}

if ($TrumpJob -or $BidenJob) {
    Write-Host "[setup] Waiting for training jobs to finish …" -ForegroundColor Yellow
    if ($TrumpJob) {
        $TrumpJob.WaitForExit()
        if ($TrumpJob.ExitCode -ne 0) { Write-Warning "Trump training exited with code $($TrumpJob.ExitCode)" }
    }
    if ($BidenJob) {
        $BidenJob.WaitForExit()
        if ($BidenJob.ExitCode -ne 0) { Write-Warning "Biden training exited with code $($BidenJob.ExitCode)" }
    }
}

# ── 7. Launch debate ───────────────────────────────────────────────────────────

Write-Host "`n[setup] Setup complete — launching debate …`n" -ForegroundColor Green
python (Join-Path $Root "scripts\start_debate.py") --persona both --tts auto --turns 5

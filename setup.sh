#!/usr/bin/env bash
# setup.sh — end-to-end project bootstrap for Linux / macOS
# Usage:  bash setup.sh
# Runs:   venv creation → install → data collection → preprocessing →
#         voice prep → parallel training → debate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Virtual environment ─────────────────────────────────────────────────────

if [ ! -d "$ROOT/.venv" ]; then
    echo -e "\n[setup] Creating virtual environment …"
    python3 -m venv "$ROOT/.venv"
fi

echo "[setup] Activating virtual environment …"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# ── 2. Dependencies ────────────────────────────────────────────────────────────

echo -e "\n[setup] Installing PyTorch (CUDA 12.8) …"
pip install --quiet torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "[setup] Installing remaining requirements …"
pip install --quiet -r "$ROOT/requirements.txt"

# ── 3. Data collection ─────────────────────────────────────────────────────────

echo -e "\n[setup] Collecting raw data …"
python "$ROOT/scripts/collect_data.py"

# ── 4. Preprocessing ───────────────────────────────────────────────────────────

echo -e "\n[setup] Preprocessing data …"
python "$ROOT/scripts/preprocess_data.py"

# ── 5. Voice reference prep ────────────────────────────────────────────────────

echo -e "\n[setup] Preparing voice references …"
python "$ROOT/scripts/prepare_voices.py"

# ── 6. Parallel training ───────────────────────────────────────────────────────

echo -e "\n[setup] Starting Trump and Biden training in parallel …"
python "$ROOT/scripts/train_trump.py" &
TRUMP_PID=$!

python "$ROOT/scripts/train_biden.py" &
BIDEN_PID=$!

echo "[setup] Waiting for both training jobs to finish …"
wait $TRUMP_PID || echo "[WARNING] Trump training exited with non-zero status"
wait $BIDEN_PID || echo "[WARNING] Biden training exited with non-zero status"

# ── 7. Launch debate ───────────────────────────────────────────────────────────

echo -e "\n[setup] Setup complete — launching debate …\n"
python "$ROOT/scripts/start_debate.py" --persona both --tts auto --turns 5

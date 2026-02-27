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

if [ -f "$ROOT/data/raw/biden/biden_debate_2020_1.txt" ] && \
   [ -f "$ROOT/data/raw/trump/trump_debate_2020_1.txt" ]; then
    echo -e "\n[setup] Raw data already exists — skipping collection."
else
    echo -e "\n[setup] Collecting raw data …"
    python "$ROOT/scripts/collect_data.py"
fi

# ── 4. Preprocessing ───────────────────────────────────────────────────────────

if [ -f "$ROOT/data/processed/biden_train.txt" ] && \
   [ -f "$ROOT/data/processed/trump_train.txt" ]; then
    echo -e "\n[setup] Processed data already exists — skipping preprocessing."
else
    echo -e "\n[setup] Preprocessing data …"
    python "$ROOT/scripts/preprocess_data.py"
fi

# ── 5. Voice reference prep ────────────────────────────────────────────────────

if [ -f "$ROOT/data/voices/biden_reference.wav" ] && \
   [ -f "$ROOT/data/voices/trump_reference.wav" ]; then
    echo -e "\n[setup] Voice references already exist — skipping voice prep."
else
    echo -e "\n[setup] Preparing voice references …"
    python "$ROOT/scripts/prepare_voices.py" || {
        echo "[WARNING] Voice preparation encountered an error (non-fatal)."
        echo "          TTS will use the default voice. Run prepare_voices.py --mode import later."
    }
fi

# ── 6. Parallel training ───────────────────────────────────────────────────────

TRUMP_PID=""
BIDEN_PID=""

if [ -f "$ROOT/models/trump/model.safetensors" ]; then
    echo -e "\n[setup] Trump model already exists — skipping training."
else
    echo -e "\n[setup] Starting Trump training in background …"
    python "$ROOT/scripts/train_trump.py" &
    TRUMP_PID=$!
fi

if [ -f "$ROOT/models/biden/model.safetensors" ]; then
    echo -e "\n[setup] Biden model already exists — skipping training."
else
    echo -e "\n[setup] Starting Biden training in background …"
    python "$ROOT/scripts/train_biden.py" &
    BIDEN_PID=$!
fi

if [ -n "$TRUMP_PID" ] || [ -n "$BIDEN_PID" ]; then
    echo "[setup] Waiting for training jobs to finish …"
    [ -n "$TRUMP_PID" ] && { wait "$TRUMP_PID" || echo "[WARNING] Trump training exited with non-zero status"; }
    [ -n "$BIDEN_PID" ] && { wait "$BIDEN_PID" || echo "[WARNING] Biden training exited with non-zero status"; }
fi

# ── 7. Launch debate ───────────────────────────────────────────────────────────

echo -e "\n[setup] Setup complete — launching debate …\n"
python "$ROOT/scripts/start_debate.py" --persona both --tts auto --turns 5

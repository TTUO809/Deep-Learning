#!/bin/bash
set -e

# --- Config ------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONDA_ENV="DL_LAB6"  # Can be overridden by your own conda env name

WANDB_NAME="lab6_${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
# ------------------------------------------------------------


# --------------- FLAGS ---------------
DO_TRAIN=true
DO_EVAL=true
DO_DENOISE=true
DO_ABLATION=true
WANDB_NAME="lab6_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-train)    DO_TRAIN=false;    shift ;;
        --no-eval)     DO_EVAL=false;     shift ;;
        --no-denoise)  DO_DENOISE=false;  shift ;;
        --no-ablation) DO_ABLATION=false; shift ;;
        --name)        WANDB_NAME="$2";   shift 2 ;;
        *)
            echo "[run.sh] Unknown argument: $1"
            echo "Usage: $0 [--no-train] [--no-eval] [--no-denoise] [--no-ablation] [--name RUN_NAME]"
            exit 1
            ;;
    esac
done

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SRC"

# --------------- STEP 1: TRAIN ---------------
if [ "$DO_TRAIN" = true ]; then
    if [ ! -f "$CKPT_DIR/last.pt" ]; then
        echo "[run.sh] Starting training from scratch (wandb name: $WANDB_NAME)..."
        python train.py \
            --epochs 200 \
            --batch_size 16 \
            --num_workers 4 \
            --amp \
            --save_every 10 \
            --wandb \
            --wandb_name "$WANDB_NAME"
    else
        # Extract the wandb run_id saved in the checkpoint so we resume the same run
        WANDB_RUN_ID=$(python - <<'EOF'
import sys, torch
try:
    ck = torch.load("../checkpoints/last.pt", map_location="cpu")
    print(ck.get("wandb_run_id", ""))
except Exception:
    print("")
EOF
)
        if [ -n "$WANDB_RUN_ID" ]; then
            echo "[run.sh] Found $CKPT_DIR/last.pt -- resuming (wandb run: $WANDB_RUN_ID)..."
            python train.py \
                --epochs 200 \
                --batch_size 16 \
                --num_workers 4 \
                --amp \
                --save_every 10 \
                --resume "$CKPT_DIR/last.pt" \
                --wandb \
                --wandb_id "$WANDB_RUN_ID"
        else
            echo "[run.sh] Found $CKPT_DIR/last.pt -- resuming (no wandb run_id in checkpoint, starting fresh wandb run)..."
            python train.py \
                --epochs 200 \
                --batch_size 16 \
                --num_workers 4 \
                --amp \
                --save_every 10 \
                --resume "$CKPT_DIR/last.pt" \
                --wandb
        fi
    fi
else
    echo "[run.sh] Skipping train."
fi

# --------------- STEP 2: EVALUATE ---------------
if [ "$DO_EVAL" = true ]; then
    echo ""
    echo "[run.sh] Evaluating best checkpoint with DDIM..."
    python evaluate.py \
        --ckpt "$CKPT_DIR/last.pt" \
        --sampler ddim \
        --steps 100
else
    echo "[run.sh] Skipping eval."
fi

# --------------- STEP 3: DENOISING PROCESS ---------------
if [ "$DO_DENOISE" = true ]; then
    echo ""
    echo "[run.sh] Generating denoising-process figure..."
    python denoise_process.py \
        --ckpt "$CKPT_DIR/last.pt" \
        --scheduler ddpm \
        --snapshots 8
else
    echo "[run.sh] Skipping denoise process."
fi

# --------------- STEP 4: ABLATION (for report extra discussion) ---------------
if [ "$DO_ABLATION" = true ]; then
    echo ""
    echo "[run.sh] Running ablation samplers for report..."
    for SAMPLER in ddpm ddim; do
        python evaluate.py \
            --ckpt "$CKPT_DIR/last.pt" \
            --sampler "$SAMPLER" \
            --steps 100 \
            --tag "$SAMPLER"
    done
    for SCALE in 1 5; do
        python evaluate.py \
            --ckpt "$CKPT_DIR/last.pt" \
            --sampler ddim_guided \
            --steps 100 \
            --guidance_scale "$SCALE" \
            --tag "guided_s${SCALE}"
    done
else
    echo "[run.sh] Skipping ablation."
fi

echo ""
echo "[run.sh] All done! Check results/ and images/ for outputs."

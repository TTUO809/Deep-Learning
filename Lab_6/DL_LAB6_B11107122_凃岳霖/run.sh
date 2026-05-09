#!/bin/bash
set -e

# --------------- Config ---------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONDA_ENV="DL_LAB6"  # Can be overridden by your own conda env name

WANDB_NAME="lab6_${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
LAST_CKPT="$CKPT_DIR/last.pt"

# --------------- FLAGS ---------------
DO_TRAIN=true
DO_EVAL=true
DO_DENOISE=true
DO_ABLATION=true

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

# --------------- ENV ---------------

##### 註：如果你已經手動啟動了環境，或者沒有使用 Conda，可以直接將以下兩行註解掉 (#) #####
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SRC"

# --------------- STEP 1: TRAIN ---------------
if [ "$DO_TRAIN" = true ]; then
    TRAIN_BASE_ARGS=(
        --epochs 200
        --batch_size 16
        --num_workers 4
        --amp
        --save_every 10
        --wandb
    )
    if [ ! -f "$LAST_CKPT" ]; then
        echo "[run.sh] Starting training from scratch (wandb name: $WANDB_NAME)..."
        python train.py "${TRAIN_BASE_ARGS[@]}" \
                    --wandb_name "$WANDB_NAME"
    else
        # Extract the wandb run_id saved in the checkpoint so we resume the same run
        WANDB_RUN_ID=$(python -c "
import torch
try:
    ck = torch.load('${LAST_CKPT}', map_location='cpu')
    print(ck.get('wandb_run_id', ''))
except Exception:
    print('')
")
        if [ -n "$WANDB_RUN_ID" ]; then
            echo "[run.sh] Found $LAST_CKPT -- resuming (wandb run: $WANDB_RUN_ID)..."
            python train.py "${TRAIN_BASE_ARGS[@]}" \
                        --resume "$LAST_CKPT" \
                        --wandb_id "$WANDB_RUN_ID"
        else
            echo "[run.sh] Found $LAST_CKPT -- resuming (no wandb run_id in checkpoint, starting fresh wandb run)..."
            python train.py "${TRAIN_BASE_ARGS[@]}" \
                        --resume "$LAST_CKPT"
        fi
    fi
else
    echo "[run.sh] Skipping train."
fi

# --------------- STEP 2: EVALUATE ---------------
if [ "$DO_EVAL" = true ]; then
    echo ""
    echo "[run.sh] Evaluating best checkpoint with DDIM + classifier guidance (s=1.0)..."
    python evaluate.py \
        --ckpt "$LAST_CKPT" \
        --sampler ddim_guided \
        --steps 100 \
        --guidance_scale 1.0
else
    echo "[run.sh] Skipping eval."
fi

# --------------- STEP 3: DENOISING PROCESS ---------------
if [ "$DO_DENOISE" = true ]; then
    echo ""
    echo "[run.sh] Generating denoising-process figure..."
    python denoise_process.py \
        --ckpt "$LAST_CKPT" \
        --scheduler ddpm \
        --snapshots 8
else
    echo "[run.sh] Skipping denoise process."
fi

# --------------- STEP 4: ABLATION (for report extra discussion) ---------------
if [ "$DO_ABLATION" = true ]; then
    echo ""
    echo "[run.sh] Running ablation samplers for report..."

    # 4-A: Sampler comparison (DDPM@100 vs DDIM@100)
    for SAMPLER in ddpm ddim; do
        python evaluate.py \
            --ckpt "$LAST_CKPT" \
            --sampler "$SAMPLER" \
            --steps 100 \
            --tag "$SAMPLER"
    done

    # 4-B: DDPM full 1000 steps (baseline reference)
    python evaluate.py \
        --ckpt "$LAST_CKPT" \
        --sampler ddpm \
        --steps 1000 \
        --tag "ddpm_1000"

    # 4-C: DDIM step count sweep (speed-quality tradeoff)
    for STEPS in 25 50 200; do
        python evaluate.py \
            --ckpt "$LAST_CKPT" \
            --sampler ddim \
            --steps "$STEPS" \
            --tag "ddim_${STEPS}"
    done

    # 4-D: Guidance scale sweep (integer + fine-grained around peak)
    for SCALE in {1..5}; do
        python evaluate.py \
            --ckpt "$LAST_CKPT" \
            --sampler ddim_guided \
            --steps 100 \
            --guidance_scale "$SCALE" \
            --tag "guided_s${SCALE}"
    done
    for SCALE in 1.5 2.5; do
        python evaluate.py \
            --ckpt "$LAST_CKPT" \
            --sampler ddim_guided \
            --steps 100 \
            --guidance_scale "$SCALE" \
            --tag "guided_s${SCALE}"
    done

    # 4-E: EMA vs raw weights (training stabilization effect)
    python evaluate.py \
        --ckpt "$LAST_CKPT" \
        --sampler ddim \
        --steps 100 \
        --no_ema \
        --tag "no_ema"
else
    echo "[run.sh] Skipping ablation."
fi

echo ""
echo "[run.sh] All done! Check results/ and images/ for outputs."

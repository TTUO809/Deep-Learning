#!/bin/bash
set -e

# ─── Config ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDENT_ID="B11107122"
CONDA_ENV="dlp_lab5"
SAVE_DIR="./results/task1_${TIMESTAMP}"
DEMO_DIR="./demo_videos/task1_${TIMESTAMP}"
WANDB_NAME="task1_${TIMESTAMP}"
# ───────────────────────────────────────────────────────────

# ─── Parse arguments ───────────────────────────────────────
DO_RECORD=true
DO_EVAL=true
MODEL_PATH=""

for arg in "$@"; do
  case "$arg" in
    --record-only) DO_EVAL=false ;;
    --eval-only)   DO_RECORD=false ;;
    *)             if [ -f "$arg" ]; then MODEL_PATH="$arg"; fi ;;
  esac
done
if [ -n "$MODEL_PATH" ]; then
  DEMO_DIR="./demo_videos/$(basename "$(dirname "$MODEL_PATH")")"
fi
# ───────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "========================================"
echo " Task 1: CartPole Vanilla DQN"
echo "========================================"

START=$(date +%s)
trap '
  END=$(date +%s); TOTAL=$(( END - START ))
  HOURS=$(( TOTAL / 3600 )); MINUTES=$(( (TOTAL % 3600) / 60 )); SECS=$(( TOTAL % 60 ))
  printf "\n========================================\n Interrupted after %02dh %02dm %02ds\n========================================\n" $HOURS $MINUTES $SECS
  exit 130
' INT

# ── 1. 訓練 (若已指定既有 model 則跳過) ──────────────────────
if [ -n "$MODEL_PATH" ]; then
  BEST_PT="$MODEL_PATH"
  echo "[1/3] Skipping training; using existing model: $BEST_PT"
else
  echo "[1/3] Training..."
  python dqn_task1.py \
    --env CartPole-v1 \
    --wandb-run-name $WANDB_NAME \
    --memory-size 50000 \
    --lr 0.001 \
    --epsilon-decay 0.999 \
    --target-update-frequency 500 \
    --replay-start-size 1000 \
    --episodes 1000 \
    --save-dir $SAVE_DIR
  BEST_PT="$SAVE_DIR/best_model.pt"
fi

# ── 2. 評估 best_model (20 seeds) ──────────────────────────
echo ""
if $DO_EVAL; then
  echo "[2/3] Evaluating best_model.pt (20 seeds)..."
  python test_model_task1.py \
    --env_name CartPole-v1 \
    --model-path $BEST_PT
else
  echo "[2/3] Skipping evaluation (--record-only)"
fi

# ── 3. 錄製 Demo 影片 ──────────────────────────────────────
echo ""
if $DO_RECORD; then
  echo "[3/3] Recording demo video..."
  python test_model_task1.py \
    --env_name CartPole-v1 \
    --model-path $BEST_PT \
    --record-video \
    --output-dir $DEMO_DIR
else
  echo "[3/3] Skipping recording (--eval-only)"
fi

# ── 複製為繳交命名 ─────────────────────────────────────────
SUBMIT="LAB5_${STUDENT_ID}_task1.pt"
cp $BEST_PT ./submission/$SUBMIT
END=$(date +%s)
TOTAL_SECONDS=$(( END - START ))

HOURS=$(( TOTAL_SECONDS / 3600 ))
MINUTES=$(( (TOTAL_SECONDS % 3600) / 60 ))
SECS=$(( TOTAL_SECONDS % 60 ))

echo ""
echo "========================================"
echo " Submission snapshot: $SUBMIT"
if $DO_RECORD; then
  echo " Demo videos: $DEMO_DIR"
fi
printf " Task 1 completed in %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
echo "========================================"

#!/bin/bash
set -e

# ─── Config ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDENT_ID="B11107122"
CONDA_ENV="dlp_lab5"
SAVE_DIR="./results_task1_${TIMESTAMP}"
DEMO_DIR="./demo_videos/task1_${TIMESTAMP}"
WANDB_NAME="task1_${TIMESTAMP}"
# ───────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "========================================"
echo " Task 1: CartPole Vanilla DQN"
echo "========================================"

START=$(date +%s)

# ── 1. 訓練 ────────────────────────────────────────────────
echo "[1/3] Training..."
python dqn.py \
  --env CartPole-v1 \
  --wandb-run-name $WANDB_NAME \
  --memory-size 50000 \
  --lr 0.001 \
  --epsilon-decay 0.999 \
  --target-update-frequency 500 \
  --replay-start-size 1000 \
  --episodes 1000 \
  --save-dir $SAVE_DIR

# ── 2. 評估 best_model (20 seeds) ──────────────────────────
BEST_PT="$SAVE_DIR/best_model.pt"
echo ""
echo "[2/3] Evaluating best_model.pt (20 seeds)..."
python test_model.py \
  --env_name CartPole-v1 \
  --model-path $BEST_PT

# ── 3. 錄製 Demo 影片 ──────────────────────────────────────
echo ""
echo "[3/3] Recording demo video..."
python test_model.py \
  --env_name CartPole-v1 \
  --model-path $BEST_PT \
  --record-video \
  --output-dir $DEMO_DIR

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
echo " Demo videos: $DEMO_DIR"
printf " Task 1 completed in %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
echo "========================================"

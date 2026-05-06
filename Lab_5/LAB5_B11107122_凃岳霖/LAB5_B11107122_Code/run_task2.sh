#!/bin/bash
set -e

# ─── Config ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDENT_ID="B11107122"
CONDA_ENV="dlp_lab5"  # Can be overridden by your own conda env name
SAVE_DIR="./results/task2_${TIMESTAMP}"
DEMO_DIR="./demo_videos/task2_${TIMESTAMP}"
WANDB_NAME="task2_${TIMESTAMP}"

SUBMIT_ROOT=".."
SUBMIT="LAB5_${STUDENT_ID}_task2.pt"
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
# if [ -n "$MODEL_PATH" ]; then
#   DEMO_DIR="./demo_videos/$(basename "$(dirname "$MODEL_PATH")")"
# fi
# ───────────────────────────────────────────────────────────

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "========================================"
echo " Task 2: Pong Vanilla DQN"
echo "========================================"

# ── 設置中斷處理 ────────────────────────────────────────────

START=$(date +%s)
trap '
  sleep 2
  
  END=$(date +%s); TOTAL=$(( END - START ))
  HOURS=$(( TOTAL / 3600 )); MINUTES=$(( (TOTAL % 3600) / 60 )); SECS=$(( TOTAL % 60 ))
  
  printf "\n"
  printf "========================================\n"
  printf " Interrupted by user\n"
  printf " Elapsed: %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
  printf "========================================\n"
  
  exit 130
' INT

# ── 1. 訓練 (若已指定既有 model 則跳過) ──────────────────────
set +e  # 暫時關閉錯誤即刻退出，確保 Ctrl+C 能正確傳給 trap
if [ -n "$MODEL_PATH" ]; then
  BEST_PT="$MODEL_PATH"
  echo "[1/3] Skipping training; using existing model: $BEST_PT"
else
  echo "[1/3] Training..."
  python dqn_task2.py \
    --env ALE/Pong-v5 \
    --wandb-run-name $WANDB_NAME \
    --memory-size 50000 \
    --lr 0.0001 \
    --epsilon-decay 0.99998 \
    --epsilon-min 0.01 \
    --target-update-frequency 1000 \
    --replay-start-size 10000 \
    --episodes 6000 \
    --save-dir $SAVE_DIR
  BEST_PT="$SAVE_DIR/best_model.pt"
fi

set -e

# ── 2. 評估 best_model (20 seeds) ──────────────────────────
echo ""
if $DO_EVAL; then
  echo "[2/3] Evaluating best_model.pt (20 seeds)..."
  python test_model_task2.py \
    --env_name ALE/Pong-v5 \
    --model-path $BEST_PT
else
  echo "[2/3] Skipping evaluation (--record-only)"
fi

# ── 3. 錄製 Demo 影片 ──────────────────────────────────────
echo ""
if $DO_RECORD; then
  echo "[3/3] Recording demo video..."
  python test_model_task2.py \
    --env_name ALE/Pong-v5 \
    --model-path $BEST_PT \
    --record-video \
    --output-dir $DEMO_DIR
else
  echo "[3/3] Skipping recording (--eval-only)"
fi

# ── 複製為繳交命名 ─────────────────────────────────────────
echo ""
echo "Copying snapshot to submission filename..."

if [ -f "$BEST_PT" ]; then
  if [ ! -f "$SUBMIT_ROOT/$SUBMIT" ]; then
    cp "$BEST_PT" "$SUBMIT_ROOT/$SUBMIT"
    echo "  Copied Best Model to: $SUBMIT_ROOT/$SUBMIT"
  else
    echo "  [Skip] $SUBMIT_ROOT/$SUBMIT already exists, will not overwrite."
  fi
else
  echo "  [Error] Best model not found at $BEST_PT"
fi

END=$(date +%s)
TOTAL_SECONDS=$(( END - START ))

HOURS=$(( TOTAL_SECONDS / 3600 ))
MINUTES=$(( (TOTAL_SECONDS % 3600) / 60 ))
SECS=$(( TOTAL_SECONDS % 60 ))

echo ""
echo "========================================"
if $DO_RECORD; then
  echo " Demo videos: $DEMO_DIR"
fi
printf " Task 2 completed in %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
echo "========================================"

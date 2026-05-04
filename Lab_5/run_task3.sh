#!/bin/bash
set -e

# ─── Config ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDENT_ID="B11107122"
CONDA_ENV="dlp_lab5"
SAVE_DIR="./results/task3_${TIMESTAMP}"
DEMO_DIR="./demo_videos/task3_${TIMESTAMP}"
WANDB_NAME="task3_${TIMESTAMP}"
MILESTONES=(600000 1000000 1500000 2000000 2500000)
# ───────────────────────────────────────────────────────────

# ─── Parse arguments ───────────────────────────────────────
DO_RECORD=true
DO_EVAL=true
SAVE_DIR_ARG=""
SPECIFIED_MILESTONE=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --record-only) DO_EVAL=false ;;
    --eval-only)   DO_RECORD=false ;;
    --milestone)   SPECIFIED_MILESTONE="$2"; shift ;;
    *)             if [ -d "$1" ] && [ -f "$1/best_model.pt" ]; then SAVE_DIR_ARG="$1"; fi ;;
  esac
  shift
done

# 如果有指定里程碑，就覆蓋預設陣列
if [ -n "$SPECIFIED_MILESTONE" ]; then
  MILESTONES=($SPECIFIED_MILESTONE)
  echo "[Config] Specific milestone specified: $SPECIFIED_MILESTONE"
fi

if [ -n "$SAVE_DIR_ARG" ]; then
  SAVE_DIR="$SAVE_DIR_ARG"
  DEMO_DIR="./demo_videos/$(basename "$SAVE_DIR_ARG")"
fi
# ───────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "========================================"
echo " Task 3: Enhanced DQN (DDQN + PER + Multi-step n=3)"
echo "========================================"

START=$(date +%s)
trap '
  END=$(date +%s); TOTAL=$(( END - START ))
  HOURS=$(( TOTAL / 3600 )); MINUTES=$(( (TOTAL % 3600) / 60 )); SECS=$(( TOTAL % 60 ))
  INTERRUPT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  printf "\n========================================\n"
  printf " [!] Script interrupted by user at %s\n" "$INTERRUPT_TIME"
  printf " Elapsed time: %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
  printf "========================================\n"
  exit 130
' INT

# ── 1. 訓練 (若已指定既有 save dir 則跳過) ────────────────
set +e  # 暫時關閉錯誤即刻退出，確保 Ctrl+C 能正確傳給 trap

if [ -n "$SAVE_DIR_ARG" ]; then
  SAVE_DIR="$SAVE_DIR_ARG"
  echo "[1/4] Skipping training; using existing save dir: $SAVE_DIR"
else
  echo "[1/4] Training..."
  python dqn.py \
    --env ALE/Pong-v5 \
    --wandb-run-name $WANDB_NAME \
    --lr 0.00025 \
    --epsilon-decay 0.99999 \
    --epsilon-min 0.01 \
    --target-update-frequency 1000 \
    --replay-start-size 10000 \
    --use-double \
    --use-per \
    --n-step 3 \
    --per-alpha 0.5 \
    --per-beta-steps 2000000 \
    --episodes 1500 \
    --save-dir $SAVE_DIR
fi

set -e  # 訓練結束後恢復保護機制

BEST_PT="$SAVE_DIR/best_model.pt"

# ── 2. 評估各 Milestone Snapshot ───────────────────────────
echo ""
if $DO_EVAL; then
  echo "[2/4] Evaluating milestone snapshots..."
  for STEPS in "${MILESTONES[@]}"; do
    PT="$SAVE_DIR/model_${STEPS}.pt"
    if [ -f "$PT" ]; then
      echo ""
      echo "--- Snapshot: ${STEPS} env steps ---"
      python test_model.py \
        --task 3 \
        --env_name ALE/Pong-v5 \
        --model-path $PT \
        --env-steps $STEPS
    else
      echo "WARNING: $PT not found (milestone not reached during training)"
    fi
  done
else
  echo "[2/4] Skipping milestone evaluation (--record-only)"
fi

# ── 3. 評估 best_model (20 seeds) ──────────────────────────
echo ""
if $DO_EVAL; then
  echo "[3/4] Evaluating best_model.pt (20 seeds)..."
  python test_model.py \
    --task 3 \
    --env_name ALE/Pong-v5 \
    --model-path $BEST_PT
  
else
  echo "[3/4] Skipping best model evaluation (--record-only)"
fi

# ── 4. 錄製 Demo 影片 ──────────────────────────────────────
echo ""
if $DO_RECORD; then
  echo "[4/4] Recording demo video..."
  python test_model.py \
    --task 3 \
    --env_name ALE/Pong-v5 \
    --model-path $BEST_PT \
    --record-video \
    --output-dir $DEMO_DIR
else
  echo "[4/4] Skipping recording (--eval-only)"
fi

# ── 複製為繳交命名 ─────────────────────────────────────────
echo ""
echo "Copying snapshots to submission filenames..."
for STEPS in "${MILESTONES[@]}"; do
  PT="$SAVE_DIR/model_${STEPS}.pt"
  if [ -f "$PT" ]; then
    DEST="LAB5_${STUDENT_ID}_task3_${STEPS}.pt"
    cp $PT ./submission/$DEST
    echo "  $DEST"
  fi
done
SUBMIT_BEST="LAB5_${STUDENT_ID}_task3_best.pt"
cp $BEST_PT ./submission/$SUBMIT_BEST
echo "  $SUBMIT_BEST"

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
printf " Task 3 completed in %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
echo "========================================"

#!/bin/bash
set -e

# ─── Config ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDENT_ID="B11107122"
CONDA_ENV="dlp_lab5"  # Can be overridden by your own conda env name
SAVE_DIR="./results/task3_${TIMESTAMP}"
DEMO_DIR="./demo_videos/task3_${TIMESTAMP}"
WANDB_NAME="task3_${TIMESTAMP}"
MILESTONES=(600000 1000000 1500000 2000000 2500000)

SUBMIT_ROOT=".."
SUBMIT="LAB5_${STUDENT_ID}_task3_best.pt"
# ───────────────────────────────────────────────────────────

# ─── Parse arguments ───────────────────────────────────────
DO_RECORD=true
DO_EVAL=true
SAVE_DIR_ARG=""
SPECIFIED_MILESTONE=""
SPECIFIC_PT_FILE=""
FOUND_BEST_PT=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --record-only) DO_EVAL=false ;;
    --eval-only)   DO_RECORD=false ;;
    --milestone)   SPECIFIED_MILESTONE="$2"; shift ;;
    *)             
      # 判斷傳入的是否為資料夾
      if [ -d "$1" ]; then 
        # 檢查資料夾內是否存在 best_model.pt 或已經命名的 SUBMIT 檔案
        if [ -f "$1/best_model.pt" ]; then
          SAVE_DIR_ARG="$1"
          FOUND_BEST_PT="$1/best_model.pt"
        elif [ -f "$1/$SUBMIT" ]; then
          SAVE_DIR_ARG="$1"
          FOUND_BEST_PT="$1/$SUBMIT"
        fi
      # 判斷傳入的是否為單一 .pt 檔案
      elif [ -f "$1" ] && [[ "$1" == *.pt ]]; then
        SPECIFIC_PT_FILE="$1"
      fi 
      ;;
  esac
  shift
done

# 如果有指定里程碑，就覆蓋預設陣列
if [ -n "$SPECIFIED_MILESTONE" ]; then
  MILESTONES=($SPECIFIED_MILESTONE)
  echo "[Config] Specific milestone specified: $SPECIFIED_MILESTONE"
fi

# if [ -n "$SAVE_DIR_ARG" ]; then
#   SAVE_DIR="$SAVE_DIR_ARG"
#   DEMO_DIR="./demo_videos/$(basename "$SAVE_DIR_ARG")"
# fi
# ───────────────────────────────────────────────────────────

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "========================================"
echo " Task 3: Enhanced DQN (DDQN + PER + Multi-step n=3)"
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

# ── 1. 訓練 (若已指定既有 save dir 則跳過) ────────────────
set +e  # 暫時關閉錯誤即刻退出，確保 Ctrl+C 能正確傳給 trap

EXTRACTED_STEPS=0

if [ -n "$SAVE_DIR_ARG" ]; then
  SAVE_DIR="$SAVE_DIR_ARG"
  BEST_PT="$FOUND_BEST_PT"
  echo "[1/4] Skipping training; using existing save dir: $SAVE_DIR"
  echo "      Found best model at: $BEST_PT"
elif [ -n "$SPECIFIC_PT_FILE" ]; then
  BEST_PT="$SPECIFIC_PT_FILE"
  MILESTONES=() # 清空 Milestone，因為我們只要測試指定的單一檔案
  echo "[1/4] Skipping training; using specific model file: $SPECIFIC_PT_FILE"
  
  # 若檔名包含數字（例如 _2500000.pt），使用正規表示式提取步數，以利後續計分
  if [[ "$SPECIFIC_PT_FILE" =~ _([0-9]+)\.pt$ ]]; then
    EXTRACTED_STEPS="${BASH_REMATCH[1]}"
    echo "      Extracted env-steps ($EXTRACTED_STEPS) from filename."
  fi
else
  echo "[1/4] Training..."
  python dqn_task3.py \
    --env ALE/Pong-v5 \
    --wandb-run-name $WANDB_NAME \
    --lr 0.0001 \
    --epsilon-decay 0.99995 \
    --epsilon-min 0.01 \
    --target-update-frequency 10000 \
    --replay-start-size 10000 \
    --use-double \
    --use-per \
    --n-step 3 \
    --per-alpha 0.6 \
    --per-beta-steps 600000 \
    --episodes 1500 \
    --save-dir $SAVE_DIR

  BEST_PT="$SAVE_DIR/best_model.pt"
fi

set -e  # 訓練結束後恢復保護機制

# ── 2. 評估各 Milestone Snapshot ───────────────────────────
echo ""
if $DO_EVAL; then
  if [ ${#MILESTONES[@]} -gt 0 ]; then
    echo "[2/4] Evaluating milestone snapshots..."
    for STEPS in "${MILESTONES[@]}"; do
      # 支援兩種檔案命名格式
      PT_NORMAL="$SAVE_DIR/model_${STEPS}.pt"
      PT_SUBMIT="$SAVE_DIR/LAB5_${STUDENT_ID}_task3_${STEPS}.pt"
      
      PT_TO_USE=""
      if [ -f "$PT_NORMAL" ]; then
        PT_TO_USE="$PT_NORMAL"
      elif [ -f "$PT_SUBMIT" ]; then
        PT_TO_USE="$PT_SUBMIT"
      fi

      if [ -n "$PT_TO_USE" ]; then
        echo ""
        echo "--- Snapshot: ${STEPS} env steps --- ($PT_TO_USE)"
        python test_model_task3.py \
          --task 3 \
          --env_name ALE/Pong-v5 \
          --model-path "$PT_TO_USE" \
          --env-steps $STEPS
      else
        echo "WARNING: Milestone ${STEPS} not found in $SAVE_DIR"
      fi
    done
  else
    echo "[2/4] Skipping milestone snapshots (Evaluating single specific file)"
  fi
else
  echo "[2/4] Skipping milestone evaluation (--record-only)"
fi

# ── 3. 評估 best_model / 指定的檔案 (20 seeds) ───────────────
echo ""
if $DO_EVAL; then
  if [ -n "$BEST_PT" ] && [ -f "$BEST_PT" ]; then
    echo "[3/4] Evaluating model: $BEST_PT (20 seeds)..."
    
    ENV_STEPS_PARAM=""
    if [ "$EXTRACTED_STEPS" -gt 0 ]; then
      ENV_STEPS_PARAM="--env-steps $EXTRACTED_STEPS"
    fi

    python test_model_task3.py \
      --task 3 \
      --env_name ALE/Pong-v5 \
      --model-path "$BEST_PT" \
      $ENV_STEPS_PARAM
  else
    echo "[3/4] Skipping best model evaluation (File not found)"
  fi
else
  echo "[3/4] Skipping model evaluation (--record-only)"
fi

# ── 4. 錄製 Demo 影片 ──────────────────────────────────────
echo ""
if $DO_RECORD; then
  if [ -n "$BEST_PT" ] && [ -f "$BEST_PT" ]; then
    echo "[4/4] Recording demo video..."
    
    ENV_STEPS_PARAM=""
    if [ "$EXTRACTED_STEPS" -gt 0 ]; then
      ENV_STEPS_PARAM="--env-steps $EXTRACTED_STEPS"
    fi

    python test_model_task3.py \
      --task 3 \
      --env_name ALE/Pong-v5 \
      --model-path "$BEST_PT" \
      --record-video \
      --output-dir $DEMO_DIR \
      $ENV_STEPS_PARAM
  else
    echo "[4/4] Skipping recording (Best model not found)"
  fi
else
  echo "[4/4] Skipping recording (--eval-only)"
fi

# ── 複製為繳交命名 ─────────────────────────────────────────
echo ""
# 若是直接測試繳交用的特定 .pt 檔，則略過複製的步驟
if [ -z "$SPECIFIC_PT_FILE" ]; then
  echo "Copying snapshots to submission filenames..."
  for STEPS in "${MILESTONES[@]}"; do
    PT_NORMAL="$SAVE_DIR/model_${STEPS}.pt"
    PT_SUBMIT="$SAVE_DIR/LAB5_${STUDENT_ID}_task3_${STEPS}.pt"
    
    PT_TO_USE=""
    if [ -f "$PT_NORMAL" ]; then PT_TO_USE="$PT_NORMAL"; fi
    if [ -f "$PT_SUBMIT" ]; then PT_TO_USE="$PT_SUBMIT"; fi

    if [ -n "$PT_TO_USE" ]; then
      DEST="LAB5_${STUDENT_ID}_task3_${STEPS}.pt"
      TARGET="$SUBMIT_ROOT/$DEST"
      
      # 檢查是否為同一個實體檔案 (避免自複製錯誤) 或是目標是否已存在
      if [ -f "$TARGET" ] && [ "$PT_TO_USE" -ef "$TARGET" ]; then
         echo "  [Skip] Milestone $DEST is already at the target location."
      elif [ ! -f "$TARGET" ]; then
        cp "$PT_TO_USE" "$TARGET"
        echo "  Copied Milestone: $TARGET"
      else
        echo "  [Skip] Milestone $DEST already exists, skipping."
      fi
    fi
  done

  if [ -n "$BEST_PT" ] && [ -f "$BEST_PT" ]; then
    if [ -f "$SUBMIT_ROOT/$SUBMIT" ] && [ "$BEST_PT" -ef "$SUBMIT_ROOT/$SUBMIT" ]; then
       echo "  [Skip] Best model is already at $SUBMIT_ROOT/$SUBMIT."
    elif [ ! -f "$SUBMIT_ROOT/$SUBMIT" ]; then
      cp "$BEST_PT" "$SUBMIT_ROOT/$SUBMIT"
      echo "  Copied Best Model to: $SUBMIT_ROOT/$SUBMIT"
    else
      echo "  [Skip] $SUBMIT_ROOT/$SUBMIT already exists, will not overwrite."
    fi
  else
    echo "  [Info] No best model found to copy."
  fi
else
  echo "Specific .pt file was provided. Skipping the submission copy phase."
fi

END=$(date +%s)
TOTAL_SECONDS=$(( END - START ))

HOURS=$(( TOTAL_SECONDS / 3600 ))
MINUTES=$(( (TOTAL_SECONDS % 3600) / 60 ))
SECS=$(( TOTAL_SECONDS % 60 ))

echo ""
echo "========================================"
if $DO_RECORD && [ -n "$BEST_PT" ] && [ -d "$DEMO_DIR" ]; then
  echo " Demo videos: $DEMO_DIR"
fi
printf " Task 3 completed in %02dh %02dm %02ds\n" $HOURS $MINUTES $SECS
echo "========================================"

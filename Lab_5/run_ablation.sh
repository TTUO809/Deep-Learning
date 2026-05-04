#!/bin/bash
set -e

# ============================================================
# Ablation Study: DDQN + PER + N-step on top of vanilla DQN
#
# 目的：在「完全相同」的 baseline 超參數下，逐一加入每個 enhancement，
#      隔離出每個技術對 sample efficiency 的貢獻。
#
# Baseline 超參數 = Task 2 的 vanilla DQN 設定，
# 唯一切換的是 --use-double / --use-per / --n-step。
# ============================================================

# ─── Config ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONDA_ENV="dlp_lab5"
ABLATION_ROOT="./results/ablation_${TIMESTAMP}"
EPISODES=1000          

# Task 2 / Task 3 共用 baseline 超參數（5 條消融全部共用）
# 與 run_task2.sh / run_task3.sh 完全對齊，僅變動 enhancement flags
COMMON_ARGS="\
  --env ALE/Pong-v5 \
  --lr 0.00025 \
  --epsilon-decay 0.99999 \
  --epsilon-min 0.01 \
  --target-update-frequency 1000 \
  --replay-start-size 10000 \
  --per-alpha 0.5 \
  --per-beta-steps 2000000 \
  --episodes ${EPISODES}"
# ───────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

mkdir -p $ABLATION_ROOT

# 捕捉 Ctrl+C，安全終止整個腳本並印出終止時間
trap '
  INTERRUPT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  echo -e "\n[!] Script completely interrupted by user at ${INTERRUPT_TIME}!"
  exit 130
' INT

# ─── 5 條消融條件 ─────────────────────────────────────────
# 名稱        旗標
# vanilla     (none)
# ddqn_only   --use-double
# per_only    --use-per
# nstep_only  --n-step 3
# all_three   --use-double --use-per --n-step 3

declare -A CONDITIONS=(
  ["vanilla"]=""
  ["ddqn_only"]="--use-double"
  ["per_only"]="--use-per"
  ["nstep_only"]="--n-step 3"
  ["all_three"]="--use-double --use-per --n-step 3"
)

# 固定執行順序（associative array 在 bash 是無序的）# "vanilla" 、"ddqn_only" 、"nstep_only"
ORDER=("per_only"  "all_three")

GLOBAL_START=$(date +%s)
START_TIME_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo "========================================"
echo " Ablation Study (Robust Version)"
echo " Started at: $START_TIME_STR"
echo " Output root: $ABLATION_ROOT"
echo " Episodes per run: $EPISODES"
echo " Conditions: ${ORDER[@]}"
echo "========================================"

# 暫時關閉 set -e，讓我們可以手動處理迴圈內的錯誤
set +e 

for NAME in "${ORDER[@]}"; do
  FLAGS="${CONDITIONS[$NAME]}"
  SAVE_DIR="${ABLATION_ROOT}/${NAME}"
  WANDB_NAME="ablation_${NAME}_${TIMESTAMP}"

  echo ""
  echo "----------------------------------------"
  echo " [Ablation] ${NAME}"
  echo "  flags : ${FLAGS:-<none>}"
  echo "  save  : ${SAVE_DIR}"
  echo "  wandb : ${WANDB_NAME}"
  echo "----------------------------------------"

  # 1. 斷點續傳：檢查是否已經跑完
  if [ -f "${SAVE_DIR}/best_model.pt" ]; then
    echo " ⏭️  [Skip] ${NAME} already completed (found best_model.pt). Moving to next..."
    continue
  fi

  RUN_START=$(date +%s)

  # 2. 單點故障隔離：檢查執行狀態
  if python dqn.py \
    $COMMON_ARGS \
    $FLAGS \
    --wandb-run-name $WANDB_NAME \
    --save-dir $SAVE_DIR; then

  RUN_END=$(date +%s)
  RUN_SEC=$(( RUN_END - RUN_START ))
      FINISH_TIME=$(date "+%Y-%m-%d %H:%M:%S")
      printf " ✅ [Done] %s finished at %s (Took %02dh %02dm %02ds)\n" "$NAME" "$FINISH_TIME" $((RUN_SEC/3600)) $(((RUN_SEC%3600)/60)) $((RUN_SEC%60))
  else
      FAIL_TIME=$(date "+%Y-%m-%d %H:%M:%S")
      echo " ❌ [Error] Run ${NAME} failed or was interrupted at ${FAIL_TIME}! Check the logs."
  fi

done

# 恢復 set -e
set -e

GLOBAL_END=$(date +%s)
GLOBAL_SEC=$(( GLOBAL_END - GLOBAL_START ))
END_TIME_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo "========================================"
echo " All Ablation tasks finished at: $END_TIME_STR"
printf " Total time elapsed: %02dh %02dm %02ds\n" $((GLOBAL_SEC/3600)) $(((GLOBAL_SEC%3600)/60)) $((GLOBAL_SEC%60))
echo "========================================"
#!/bin/bash
# Run monosemanticity scoring for multiple SAE configs in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"

GPU_IDS=(
  # "3"
  # "7"
  "6"
  # "5"
)

MAX_N=500000
LOG_DIR="scripts_experiments/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting Monosemanticity Jobs (Parallel)"
echo "Foundation models: ${FOUNDATION_MODELS[*]}"
echo "=========================================="

for FOUNDATION_MODEL_NAME in "${FOUNDATION_MODELS[@]}"; do
  set_foundation_model_configs "$FOUNDATION_MODEL_NAME"

  if [[ ${#CONFIGS[@]} -ne ${#GPU_IDS[@]} ]]; then
    echo "ERROR: CONFIGS and GPU_IDS must have the same length for model=$FOUNDATION_MODEL"
    exit 1
  fi

  echo ""
  echo "=========================================="
  echo "Processing foundation model: $FOUNDATION_MODEL"
  echo "Configs: ${CONFIGS[*]}"
  echo "=========================================="

  declare -a PIDS=()
  declare -a JOB_NAMES=()

  for i in "${!CONFIGS[@]}"; do
    CONFIG_NAME="${CONFIGS[$i]}"
    GPU_ID="${GPU_IDS[$i]}"
    LOG_FILE="$LOG_DIR/monosemanticity_${FOUNDATION_MODEL_NAME}_${CONFIG_NAME}.log"

    (
      set -euo pipefail
      export MSAE_CONFIG="$CONFIG_NAME"

      echo "[$(date '+%F %T')] Starting model=$FOUNDATION_MODEL config=$CONFIG_NAME on cuda:$GPU_ID"
      python -m label_assignment_strategies.individual_feature_scores.monosemanticity_score \
        -max_n "$MAX_N" \
        -a \
        --device "cuda:${GPU_ID}"
      echo "[$(date '+%F %T')] Finished model=$FOUNDATION_MODEL config=$CONFIG_NAME"
    ) >"$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=("$PID")
    JOB_NAMES+=("$CONFIG_NAME")
    echo "Launched $CONFIG_NAME on cuda:$GPU_ID (pid=$PID, log=$LOG_FILE)"
  done

  echo ""
  echo "Waiting for all jobs to complete for model=$FOUNDATION_MODEL..."

  FAILED=0
  for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    JOB_NAME="${JOB_NAMES[$i]}"

    if wait "$PID"; then
      echo "SUCCESS: $JOB_NAME"
    else
      echo "FAILED:  $JOB_NAME"
      FAILED=1
    fi
  done

  if [[ $FAILED -ne 0 ]]; then
    echo "=========================================="
    echo "One or more jobs failed for model=$FOUNDATION_MODEL"
    echo "Check logs under $LOG_DIR"
    echo "=========================================="
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "All monosemanticity jobs completed successfully"
echo "=========================================="
exit 0


#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"

# Shared args
FOUNDATION_MODEL="dinov2-base" # "ViT-L~14" or "dinov2-base"
DT=data/cc3m_${FOUNDATION_MODEL}_train_image_2820737_768.npy
DS=data/cc3m_${FOUNDATION_MODEL}_validation_image_13002_768.npy
EF=8
EPOCHS=30
MS=1820737
# tqdm behavior: 1 disables progress bars, 0 enables them
TQDM_DISABLE=1

LOG_DIR=scripts_experiments/logs
mkdir -p "$LOG_DIR"

# Format: "GPU MODEL ACTIVATION"
JOBS=(
  "1 MSAE_RW TopKReLU"
  "2 ArchMSAE_UW TopKReLU_64"
  "3 TopKSAE TopKReLU_64"
  "4 MPSAE TopKReLU_100000"
  "5 EWGSAE ReLU"
)

OVERALL_FAILED=0

# Seeds are defined in shared_variables.sh and run sequentially: all JOBS for
# one seed launch in parallel and must finish before the next seed starts. This
# keeps GPU occupancy bounded by len(JOBS) regardless of len(SEEDS).
for SEED in "${SEEDS[@]}"; do
  echo "=========================================="
  echo "Starting training pass for seed=$SEED"
  echo "=========================================="

  PIDS=()
  NAMES=()

  for JOB in "${JOBS[@]}"; do
    read -r GPU MODEL ACT <<< "$JOB"
    LOG_FILE="$LOG_DIR/train_${MODEL}_seed${SEED}.log"

    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$GPU"
      export TQDM_DISABLE="$TQDM_DISABLE"
      python -m train \
        -dt "$DT" \
        -ds "$DS" \
        --expansion_factor "$EF" \
        --epochs "$EPOCHS" \
        -m "$MODEL" \
        -a "$ACT" \
        -ms "$MS" \
        --seed "$SEED"
    ) > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=("$PID")
    NAMES+=("$MODEL")
    echo "Launched $MODEL (seed=$SEED) on GPU $GPU (pid=$PID, log=$LOG_FILE)"
  done

  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "SUCCESS: ${NAMES[$i]} (seed=$SEED)"
    else
      echo "FAILED:  ${NAMES[$i]} (seed=$SEED)"
      OVERALL_FAILED=1
    fi
  done
done

exit "$OVERALL_FAILED"

# OLD manual
# # make values configurable as variables
# # - dt
# DT=data/cc3m_ViT-L~14_train_image_2820737_768.npy
# # - ds
# DS=data/cc3m_ViT-L~14_validation_image_13002_768.npy
# # - expansion_factor
# EF=8
# # - epochs
# EPOCHS=30
# # - ms
# MS=1820737

# # OVERRIDE FOR DINO MODEL
# # - dt
# DT=data/cc3m_dinov2-base_train_image_2820737_768.npy
# # - ds
# DS=data/cc3m_dinov2-base_validation_image_13002_768.npy


# CUDA_VISIBLE_DEVICES=6 python train.py -dt $DT -ds $DS  --expansion_factor $EF --epochs $EPOCHS  -m MSAE_RW -a TopKReLU -ms $MS

# CUDA_VISIBLE_DEVICES=6 python train.py -dt $DT -ds $DS  --expansion_factor $EF --epochs $EPOCHS  -m ArchMSAE_UW -a TopKReLU_64 -ms $MS

# CUDA_VISIBLE_DEVICES=6 python train.py -dt $DT -ds $DS  --expansion_factor $EF --epochs $EPOCHS  -m TopKSAE -a TopKReLU_64 -ms $MS

# # CUDA_VISIBLE_DEVICES=6 python train.py -dt $DT -ds $DS  --expansion_factor $EF --epochs $EPOCHS  -m MultiSAE -a TopKReLU -ms $MS

# CUDA_VISIBLE_DEVICES=6 python train.py -dt $DT -ds $DS  --expansion_factor $EF --epochs $EPOCHS  -m MPSAE -a TopKReLU_100000 -ms $MS
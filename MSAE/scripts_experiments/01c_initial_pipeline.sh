#!/bin/bash
# Labelling and metrics pipeline (runs after 01_precompute_sae_data.sh)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"

# GPU device to use (default: 6, can be overridden)
GPU_DEVICE=${CUDA_VISIBLE_DEVICES:-6}

export MEAN_CENTER

echo "=========================================="
echo "Starting Initial Pipeline"
echo "Mean centering: $MEAN_CENTER"
echo "Foundation models: ${FOUNDATION_MODELS[@]}"
echo "=========================================="

for FOUNDATION_MODEL_NAME in "${FOUNDATION_MODELS[@]}"; do
    set_foundation_model_configs "$FOUNDATION_MODEL_NAME"

    echo ""
    echo "=========================================="
    echo "Processing foundation model: $FOUNDATION_MODEL"
    echo "Configs: ${CONFIGS[@]}"
    echo "=========================================="

    for CONFIG_NAME in "${CONFIGS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Processing config: $CONFIG_NAME"
        echo "=========================================="

        export MSAE_CONFIG="$CONFIG_NAME"

        echo ""
        echo "[Step 1/3] Labelling with predefined vocab..."
        python -m label_assignment_strategies.labelling_and_scoring.captions_from_predefined_vocab_with_highest_cos_sim

        echo ""
        echo "[Step 2/3] Calculating individual feature metrics..."
        python -m label_assignment_strategies.individual_feature_scores.semantic_lens_scores

        echo ""
        echo "[Step 2/3] Computing conditional activation metrics..."
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -m activations_preprocessing.compute_conditional_activation_metrics

        # echo ""
        # echo "[Step 2/3] Computing masked cosine similarity..."
        # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -m activations_preprocessing.masked_cosine_similarity

        echo ""
        echo "[Step 3/3] Creating labeling and metric summary..."
        python -m label_assignment_strategies.create_labeling_and_metric_summary_df

        echo ""
        echo "Completed processing for config: $CONFIG_NAME"
    done
done

echo ""
echo "=========================================="
echo "Initial Pipeline Completed Successfully"
echo "All foundation models processed: ${FOUNDATION_MODELS[@]}"
echo "=========================================="

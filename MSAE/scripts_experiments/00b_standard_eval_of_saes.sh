#!/bin/bash
# Initial pipeline steps for SAE feature analysis

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"


echo "=========================================="
echo "Starting Initial Pipeline"
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
        echo "Calculating SAE Metrics..."
        python -m sae_general_eval.execute_calculate_sae_metrics

        echo ""
        echo "Completed processing for config: $CONFIG_NAME"
    done
done

echo ""
echo "=========================================="
echo "Initial Pipeline Completed Successfully"
echo "All foundation models processed: ${FOUNDATION_MODELS[@]}"
echo "=========================================="


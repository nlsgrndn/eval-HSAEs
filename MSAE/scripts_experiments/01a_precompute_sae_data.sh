#!/bin/bash
# Precompute SAE data (step 1 of the initial pipeline)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"

export MEAN_CENTER

echo "=========================================="
echo "Precomputing SAE Data"
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
        echo "Precomputing SAE data..."
        python execute_precompute_sae_data.py -s ""

        echo ""
        echo "Completed precompute for config: $CONFIG_NAME"
    done
done

echo ""
echo "=========================================="
echo "SAE Data Precomputation Completed Successfully"
echo "All foundation models processed: ${FOUNDATION_MODELS[@]}"
echo "=========================================="

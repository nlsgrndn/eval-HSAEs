#!/bin/bash
# Graph creation pipeline steps

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"

export MEAN_CENTER

echo "=========================================="
echo "Starting Graph Creation Pipeline"
echo "Using GPU: $GPU_DEVICE"
echo "Foundation models: ${FOUNDATION_MODELS[@]}"
echo "Upper thresholds: ${UPPER_THRESHOLDS[@]}"
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

        for THRESHOLD in "${UPPER_THRESHOLDS[@]}"; do
            if [ "$THRESHOLD" != "default" ]; then
                export GRAPH_UPPER_THRESHOLD="$THRESHOLD"
                echo ""
                echo "=========================================="
                echo "Using upper threshold: $THRESHOLD"
                echo "=========================================="
            else
                unset GRAPH_UPPER_THRESHOLD
                echo ""
                echo "=========================================="
                echo "Using default upper thresholds"
                echo "=========================================="
            fi

            for GRAPH_TYPE in "${GRAPH_TYPES[@]}"; do
                echo ""
                echo "---- Creating graph of type: $GRAPH_TYPE ----"
                export GRAPH_TYPE="$GRAPH_TYPE"

                echo ""
                echo " Creating graph from conditional activations..."
                python -m structure_extraction.activation_behavior_methods.create_graph_from_cond_acts
            done

            echo ""
            echo "Completed processing for threshold: $THRESHOLD"
        done

        echo ""
        echo "Completed processing for config: $CONFIG_NAME"
    done
done

echo ""
echo "=========================================="
echo "Graph Creation Pipeline Completed Successfully"
echo "All foundation models processed: ${FOUNDATION_MODELS[@]}"
echo "All thresholds processed: ${UPPER_THRESHOLDS[@]}"
echo "=========================================="

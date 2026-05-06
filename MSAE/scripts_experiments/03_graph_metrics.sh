#!/bin/bash
# Graph metrics computation pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_variables.sh"

export MEAN_CENTER

for FOUNDATION_MODEL_NAME in "${FOUNDATION_MODELS[@]}"; do
    set_foundation_model_configs "$FOUNDATION_MODEL_NAME"

    echo ""
    echo "=========================================="
    echo "Processing foundation model: $FOUNDATION_MODEL"
    echo "Configs: ${CONFIGS[@]}"
    echo "=========================================="

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
        echo "---- Eval graph of type: $GRAPH_TYPE ----"
        export GRAPH_TYPE="$GRAPH_TYPE"

        for CONFIG_NAME in "${CONFIGS[@]}"; do
            echo ""
            echo "=========================================="
            echo "Processing config: $CONFIG_NAME"
            echo "=========================================="

            export MSAE_CONFIG="$CONFIG_NAME"

            echo ""
            echo "[Step 1/3] Computing hierarchy graph activation metrics..."
            python -m structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_acts_metrics

            echo ""
            echo "[Step 2/3] Computing hierarchy graph metrics based on feature hierarchicality..."
            python -m structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_metrics_based_on_feature_hierarchicality_metrics

            echo ""
            echo "[Step 3/3] Computing hierarchy graph metadata metrics..."
            python -m structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_metadata

            echo ""
            echo "[Step 4/3] Computing intra-hierarchy graph similarity metrics..."
            python -m structure_evaluation.clustering_graph_metrics.compute_intra_hierarchy_graph_sim_metrics

            echo ""
            echo "[Step 5/3] Computing feature-level metrics for hierarchy graph..."
            python -m structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_feature_level_metrics

            echo ""
            echo "[Step 6/3] Computing per-cluster depth metrics for hierarchy graph..."
            python -m structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_depth_metrics

            echo ""
            echo "[Step 6/3] Aggregating single configuration graph metrics..."
            python -m structure_evaluation.clustering_graph_metrics.aggregate_single_config_metrics

            echo ""
            echo "[Step 7/3] Aggregating single configuration raw values for all metrics..."
            python -m structure_evaluation.clustering_graph_metrics.aggregate_single_config_raw_values --metric-type all
        done
        done

        echo ""
        echo "Completed processing for threshold: $THRESHOLD"
    done
done

echo ""
echo "=========================================="
echo "Graph Metrics Computation Pipeline Completed Successfully"
echo "All foundation models processed: ${FOUNDATION_MODELS[@]}"
echo "All thresholds processed: ${UPPER_THRESHOLDS[@]}"
echo "=========================================="

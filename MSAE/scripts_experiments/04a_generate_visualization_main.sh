#!/bin/bash
# Visualization generation pipeline
set -e  # Exit on error

names=(
    "wondact_comparison_nonMC"
    # "condact_comparison"
    # "condact_vs_mcs_comparison"
    # "condact_MC_vs_nonMC_comparison"
    # "condact_comparison_nonMC"
    # "condact_vs_mcs_comparison_nonMC"
    # "condact_vs_wondact_comparison_nonMC"
)

for NAME in "${names[@]}"; do
    echo ""
    echo "[Step 1/4] Creating comparison dataframe..."
    python -m structure_evaluation.create_comparisons_data.create_comparison_dataframe -n $NAME
    echo ""

    echo ""
    echo "[Step 2/4] Visualizing comparison and exporting tables..."
    python -m structure_evaluation.create_comparisons_data.visualize_comparison --input experiments_results/$NAME.pkl --export
    echo ""

    echo ""
    echo "[Step 3/4] Creating bar chart visualization..."
    python -m structure_evaluation.create_comparisons_data.visualize_comparison_barchart --input experiments_results/$NAME.pkl --output experiments_results/${NAME}_barchart.png
    echo ""

    echo ""
    echo "[Step 4/4] Creating LaTeX structural metrics table..."
    python -m structure_evaluation.create_comparisons_data.create_latex_structural_table --input experiments_results/${NAME}_structural_metrics.pkl --output experiments_results/${NAME}_structural_metrics.tex
    echo ""

    echo ""
    echo "[Step 4/4] Creating csv structural metrics table..."
    python -m structure_evaluation.create_comparisons_data.create_statistics_csv --input experiments_results/${NAME}_structural_metrics.pkl --output experiments_results/${NAME}_structural_metrics.csv
    echo ""
done
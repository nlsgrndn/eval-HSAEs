#!/bin/bash
# Visualization generation pipeline
set -e  # Exit on error


NAME="wondact_nonMC_comparison"
# NAME="condact_comparison"
# NAME="condact_nonMC_comparison"
# NAME="mcs_comparison"
# NAME="mcs_nonMC_comparison"




# 
echo ""
echo "[Step 1/2] "
python -m structure_evaluation.create_comparisons_data.create_distribution_comparison     --comparison-name "$NAME"
echo ""

# echo ""
# echo "[Step 2/2] "
# python -m structure_evaluation.create_comparisons_data.visualize_comparison_distributions     --input experiments_results/${NAME}_distributions.pkl     --output experiments_results/raincloud_plots_${NAME}/     --plot-type raincloud     --max-points 7000
# echo ""

echo ""
echo "[Step 2/2] "
python -m structure_evaluation.create_comparisons_data.visualize_comparison_distributions     --input experiments_results/${NAME}_distributions.pkl     --output experiments_results/raincloud_plots_${NAME}/     --plot-type raincloud     --max-points 7000 --combine-plots
echo ""


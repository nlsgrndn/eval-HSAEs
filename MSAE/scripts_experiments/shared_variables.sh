#!/bin/bash

# Shared experiment variables
FOUNDATION_MODELS=("vit" "dinov2-base") #("vit" "dinov2-base")

# Seeds to iterate over in the pipeline. Each (architecture, seed) becomes its
# own MSAE_CONFIG (e.g. cc3m_vit_topk_seed43). Extend this list once additional
# seeds have been trained and added to my_config.py::model_strings.
SEEDS=(42 43 44)

set_foundation_model_configs() {
	local foundation_model="$1"
	export FOUNDATION_MODEL="$foundation_model"
	local arch_keys=(
		"cc3m_${FOUNDATION_MODEL}_hsae_v2"
		# "cc3m_${FOUNDATION_MODEL}_hsae"
		"cc3m_${FOUNDATION_MODEL}_topk"
		"cc3m_${FOUNDATION_MODEL}_archmsae_uw"
		"cc3m_${FOUNDATION_MODEL}_actmsae"
		"cc3m_${FOUNDATION_MODEL}_mpsae"
		"cc3m_${FOUNDATION_MODEL}_ewgsae"
	)
	CONFIGS=()
	for arch in "${arch_keys[@]}"; do
		for seed in "${SEEDS[@]}"; do
			CONFIGS+=("${arch}_seed${seed}")
		done
	done
}

# Initialize with the first model for backward compatibility in scripts
# that consume CONFIGS without their own model loop.
# set_foundation_model_configs "${FOUNDATION_MODELS[0]}"

GRAPH_TYPES=("wondact") # "condact" "mcs")

# UPPER_THRESHOLDS=(0.8 0.85 0.9 0.95)
UPPER_THRESHOLDS=(0.9) # 0.95)
# UPPER_THRESHOLDS=("default")

MEAN_CENTER=false

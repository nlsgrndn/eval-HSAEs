from typing import Dict
from configs.configs_definitions import FoundationModel, Dataset, SAEModel, Config
import copy
import os

# === PREDEFINED CONFIGS ===
# DEFAULT_SAE_ACTIVATIONS_SUBSET_SIZE = 0 # 0 or 200000
CONFIGS: Dict[str, Config] = {}

# Foundation models
vit_l14 = FoundationModel(name="ViT-L~14", embedding_dim=768)
dinov2_base = FoundationModel(name="dinov2-base", embedding_dim=768)

# CC3M + ViT-L/14 configs
cc3m_train = Dataset(name="cc3m", split="train", subset_str="train")
#cc3m_val = Dataset(name="cc3m", split="val", subset_str="inference")
cc3m_train_inference = Dataset(name="cc3m", split="train", subset_str="inference")
cc3m_train_graph_creation = Dataset(name="cc3m", split="train", subset_str="graph_creation")
cc3m_train_graph_evaluation = Dataset(name="cc3m", split="train", subset_str="graph_evaluation")

# cc3m_train_subset = Dataset(name="cc3m", split="train", sae_activations_dataset_subset_size=DEFAULT_SAE_ACTIVATIONS_SUBSET_SIZE)
# cc3m_val_subset = Dataset(name="cc3m", split="val", sae_activations_dataset_subset_size=DEFAULT_SAE_ACTIVATIONS_SUBSET_SIZE)

# # ImageNet configs
# imagenet_train = Dataset(name="imagenet", split="train")
# imagenet_val = Dataset(name="imagenet", split="val")

# imagenet_train_subset = Dataset(name="imagenet", split="train", sae_activations_dataset_subset_size=DEFAULT_SAE_ACTIVATIONS_SUBSET_SIZE)
# imagenet_val_subset = Dataset(name="imagenet", split="val", sae_activations_dataset_subset_size=DEFAULT_SAE_ACTIVATIONS_SUBSET_SIZE)

# Each architecture maps to {seed: timestamped_model_name}.
# Seed 42 = legacy default of train.py (runs trained without explicit --seed).
# Seeds 43, 44 = explicit --seed runs from scripts_experiments/00_train_saes.sh.
DEFAULT_SEED = 42

model_strings: Dict[str, Dict[int, str]] = {
    "cc3m_vit_topk": {
        42: "topk_k64_x8_cc3m_vit_20260119_090321",
        43: "topk_k64_x8_cc3m_vit_20260501_165831",
        44: "topk_k64_x8_cc3m_vit_20260501_170413",
    },
    "cc3m_vit_archmsae_uw": {
        42: "archmsae_uw_k64_x8_cc3m_vit_20260209_105919",
        43: "archmsae_uw_k64_x8_cc3m_vit_20260501_170255",
        44: "archmsae_uw_k64_x8_cc3m_vit_20260501_170957",
    },
    "cc3m_vit_actmsae": {
        42: "msae_rw_x8_cc3m_vit_20260118_135239",
        43: "msae_rw_x8_cc3m_vit_20260501_174053",
        44: "msae_rw_x8_cc3m_vit_20260501_174309",
    },
    "cc3m_vit_mpsae": {
        42: "mpsae_k100000_x8_cc3m_vit_20260402_130111",
        43: "mpsae_k100000_x8_cc3m_vit_20260502_020347",
        44: "mpsae_k100000_x8_cc3m_vit_20260502_020747",
    },
    # "cc3m_vit_hsae": {42: "hsae_361_16_6144_cc3m_vit"},
    "cc3m_vit_hsae_v2": {
        42: "hsae_vit_361_16_v2_cc3m",
        43: "hsae_vit_361_16_v2_1_cc3m",
        44: "hsae_vit_361_16_v2_2_cc3m",
    },
    "cc3m_vit_ewgsae": {
        42: "ewgsae_x8_cc3m_vit_20260427_160030",
        43: "ewgsae_x8_cc3m_vit_20260501_170524",
        44: "ewgsae_x8_cc3m_vit_20260501_170829",
    },
    "cc3m_dinov2-base_topk": {
        42: "topk_k64_x8_cc3m_dinov2_20260406_161501",
        43: "topk_k64_x8_cc3m_dinov2_20260501_170509",
        44: "topk_k64_x8_cc3m_dinov2_20260501_170354",
    },
    "cc3m_dinov2-base_archmsae_uw": {
        42: "archmsae_uw_k64_x8_cc3m_dinov2_20260406_161702",
        43: "archmsae_uw_k64_x8_cc3m_dinov2_20260501_171058",
        44: "archmsae_uw_k64_x8_cc3m_dinov2_20260501_171116",
    },
    "cc3m_dinov2-base_actmsae": {
        42: "msae_rw_x8_cc3m_dinov2_20260406_162608",
        43: "msae_rw_x8_cc3m_dinov2_20260501_174540",
        44: "msae_rw_x8_cc3m_dinov2_20260501_174519",
    },
    "cc3m_dinov2-base_mpsae": {
        42: "mpsae_k100000_x8_cc3m_dinov2_20260406_181311",
        43: "mpsae_k100000_x8_cc3m_dinov2_20260502_020658",
        44: "mpsae_k100000_x8_cc3m_dinov2_20260502_020633",
    },
    # "cc3m_dinov2-base_hsae": {42: "hsae_361_16_6144_cc3m_dinov2"},
    "cc3m_dinov2-base_hsae_v2": {
        42: "hsae_dino_361_16_v2_cc3m",
        43: "hsae_dino_361_16_v2_1_cc3m",
        44: "hsae_dino_361_16_v2_2_cc3m",
    },
    "cc3m_dinov2-base_ewgsae": {
        42: "ewgsae_x8_cc3m_dinov2_20260428_101858",
        43: "ewgsae_x8_cc3m_dinov2_20260501_170957",
        44: "ewgsae_x8_cc3m_dinov2_20260501_170921",
    },
}

SAE_FILTERING_STRATEGY = "all"  # "all", "same_dec_col_and_avg_sim_for_both_strategies", "top_50_percent_max_activation"
FOUNDATION_MODEL = os.environ.get('FOUNDATION_MODEL', "vit")  # "vit" or "dinov2-base"

incomplete_base_config = Config(
    sae_model=SAEModel(
        name="TODO",  # Placeholder, will be set in loop below
        foundation_model="TODO",  # Placeholder, will be set in loop below
        training_dataset=cc3m_train,
    ),
    graph_creation_dataset=cc3m_train_graph_creation,
    graph_eval_dataset=cc3m_train_graph_evaluation,
    sae_filtering_strategy=SAE_FILTERING_STRATEGY,
)

for arch_key, seed_dict in model_strings.items():
    if FOUNDATION_MODEL not in arch_key:
        continue  # Skip configs that don't match the selected foundation model
    for seed, model_name in seed_dict.items():
        config = copy.deepcopy(incomplete_base_config)
        config.sae_model.name = model_name
        config.sae_model.seed = seed
        if "_vit_" in arch_key:
            config.sae_model.foundation_model = vit_l14
        elif "_dinov2-base_" in arch_key:
            config.sae_model.foundation_model = dinov2_base
        else:
            raise ValueError(f"Unknown foundation model key for config: {arch_key}")
        CONFIGS[f"{arch_key}_seed{seed}"] = config

# === SELECT ACTIVE CONFIG ===
# Read from environment variable MSAE_CONFIG, fallback to hardcoded default.
# Legacy invocations without `_seed{N}` suffix resolve to DEFAULT_SEED so existing
# usage keeps pointing at the same model it always did.
_raw_active = os.environ.get('MSAE_CONFIG', f"cc3m_{FOUNDATION_MODEL}_hsae_v2")
if "_seed" in _raw_active:
    ACTIVE_CONFIG = _raw_active
else:
    ACTIVE_CONFIG = f"{_raw_active}_seed{DEFAULT_SEED}"
DEFAULT_CONFIG = CONFIGS[ACTIVE_CONFIG]

CONFIGS = {cfg.simple_name: cfg for cfg in CONFIGS.values()}
DEFAULT_CONFIG = CONFIGS[DEFAULT_CONFIG.simple_name]


from dataclasses import dataclass, field
import os

# possible strategies:
# percentile-based -> probably as it effectively says that a feature is active for top X% of images, but the presence of features is not at all equally distributed across features
# absolute threshold-based -> less good as different features have different activation distributions
# hybrid -> e.g. 90th percentile and absolute threshold of 0.1 activation
# relative threshold based on activation behavior of feature
class BinarizationStrategy:
    # NOTE: Don't add percentile here as theoretical considerations make clear that it doesnt make sense alone
    ABSOLUTE_THRESHOLD = "absolute_threshold"
    # HYBRID_MAX = "hybrid_max"
    # HYBRID_MIN = "hybrid_min"
    # RELATIVE = "relative"
    TOP_K = "top_k"


@dataclass
class ActivationsPreprocessingConfig:
    """Config for computing conditional activation metrics."""
    binarization_strategy: BinarizationStrategy
    binarization_kwargs: dict
    max_num_samples: int
    apply_top_k_preprocess: int
    mean_center: bool = field(default_factory=lambda: os.environ.get('MEAN_CENTER', 'false').lower() == 'true')


    @property
    def name(self) -> str:
        """Automatically generate name based on parameters."""
        # Extract binarization info
        if self.binarization_strategy == BinarizationStrategy.ABSOLUTE_THRESHOLD:
            bin_str = f"th{self.binarization_kwargs.get('absolute_threshold')}"
        elif self.binarization_strategy == BinarizationStrategy.TOP_K:
            bin_str = f"topk{self.binarization_kwargs.get('top_k')}"
        else:
            bin_str = str(self.binarization_strategy.value)
        
        # Build name parts
        parts = [bin_str]
        parts.append(f"maxNS{self.max_num_samples}")
        
        if self.mean_center:
            parts.append("MC")
        
        if self.apply_top_k_preprocess > 0:
            parts.append(f"topkPP{self.apply_top_k_preprocess}")
        
        return "_".join(parts)
    
_activation_preprocessing_configs = [
    ActivationsPreprocessingConfig(
        binarization_strategy=BinarizationStrategy.ABSOLUTE_THRESHOLD,
        binarization_kwargs={"absolute_threshold": 4},
        max_num_samples=500000,
        apply_top_k_preprocess=0
    ),
    ActivationsPreprocessingConfig(
        binarization_strategy=BinarizationStrategy.ABSOLUTE_THRESHOLD,
        binarization_kwargs={"absolute_threshold": 1},
        max_num_samples=500000,
        apply_top_k_preprocess=0
    ),
    ActivationsPreprocessingConfig(
        binarization_strategy=BinarizationStrategy.ABSOLUTE_THRESHOLD,
        binarization_kwargs={"absolute_threshold": 0.1},
        max_num_samples=500000,
        apply_top_k_preprocess=0
    ),
    ActivationsPreprocessingConfig(
        binarization_strategy=BinarizationStrategy.TOP_K,
        binarization_kwargs={"top_k": 64},
        max_num_samples=500000,
        apply_top_k_preprocess=64
    ),
    # ActivationsPreprocessingConfig(
    #     binarization_strategy=BinarizationStrategy.TOP_K,
    #     binarization_kwargs={"top_k": 64},
    #     max_num_samples=200000,
    #     mean_center=False,
    #     apply_top_k_preprocess=0
    # ),
]
ACTS_PREPROCESS_CONFIGS = {cfg.name: cfg for cfg in _activation_preprocessing_configs}
DEFAULT_ACTIVATION_PREPROCESS_CONFIG = _activation_preprocessing_configs[-1].name
 # print(f"Defined default activation preprocess config: {DEFAULT_ACTIVATION_PREPROCESS_CONFIG}")

def get_acts_preprocess_cfg(config_name: str = DEFAULT_ACTIVATION_PREPROCESS_CONFIG) -> ActivationsPreprocessingConfig:
    return ACTS_PREPROCESS_CONFIGS[config_name]
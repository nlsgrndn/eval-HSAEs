import os
from dataclasses import dataclass, field
from configs.activation_preprocessing import ActivationsPreprocessingConfig, get_acts_preprocess_cfg

@dataclass
class MCSGraphConfig:
    """Config for creating graphs from conditional activation metrics."""
    structure_type: str = "dag"  # "tree" or "dag"
    metrics_config: ActivationsPreprocessingConfig = field(default_factory=get_acts_preprocess_cfg)
    mcs_parent_given_child_threshold: float = field(default_factory=lambda: float(os.environ.get('GRAPH_UPPER_THRESHOLD', 0.8)))
    mcs_child_given_parent_threshold: float = field(default_factory=lambda: float(os.environ.get('GRAPH_UPPER_THRESHOLD', 0.8)))
    filtering_kwargs: dict = field(default_factory=dict)
    min_descendants: int = 5
    max_pct_children_rel_to_SAEwidth: float = 0.05

    
    @property
    def name(self) -> str:
        """Automatically generate name based on parameters."""
        # Start with metrics config name
        parts = [self.structure_type]

        parts+= [self.metrics_config.name]
        
        # Add filtering kwargs
        mf = self.filtering_kwargs.get('magnitude_factor')
        if mf is not None:
            parts.append(f"MF{int(mf) if isinstance(mf, (int, float)) else mf}")
        else:
            parts.append("MFNone")
        
        nh = self.filtering_kwargs.get('n_highest')
        if nh is not None:
            parts.append(f"NH{nh}")
        
        laf = self.filtering_kwargs.get('low_act_frequency')
        if laf is not None:
            # Format as fraction or None
            parts.append(f"LAF{laf}")
        

        parts.append("MCS")
        if self.mcs_parent_given_child_threshold is not None:
            pac_str = str(self.mcs_parent_given_child_threshold).replace('.', 'point')
            parts.append(f"PAC{pac_str}")
        if self.mcs_child_given_parent_threshold is not None:
            cap_str = str(self.mcs_child_given_parent_threshold).replace('.', 'point')
            parts.append(f"CAP{cap_str}")

        # Add min_descendants
        parts.append(f"MD{self.min_descendants}")

        max_pct_str = str(self.max_pct_children_rel_to_SAEwidth).replace('.', 'point')
        parts.append(f"MAXPCTCHILD{max_pct_str}")
        
        return "_".join(parts)
    


# Predefined graph configs
_GRAPH_CONFIGS_LIST = [
    MCSGraphConfig(
        metrics_config=get_acts_preprocess_cfg(),
        filtering_kwargs={
            'n_highest': None,
            'low_act_frequency': 40/1000000,
            'magnitude_factor': None
        },
    ),
]
GRAPH_CONFIGS = {cfg.name: cfg for cfg in _GRAPH_CONFIGS_LIST}
DEFAULT_GRAPH_CONFIG = _GRAPH_CONFIGS_LIST[-1].name
# print(f"Defined default graph config: {DEFAULT_GRAPH_CONFIG}")

def get_mcs_graph_config(config_name: str = DEFAULT_GRAPH_CONFIG) -> MCSGraphConfig:
    return GRAPH_CONFIGS[config_name]
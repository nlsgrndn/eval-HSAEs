"""
Shared utilities for creating and visualizing comparison dataframes.

This module contains common constants and functions used across multiple
comparison and visualization scripts to avoid code duplication.
"""

import hashlib
import re
from pathlib import Path


# ==============================================================================
# LaTeX Figure Sizing
# ==============================================================================

# Thesis textwidth in inches (512.14963pt / 72.27 pt/in)
TEXTWIDTH_INCHES = 512.14963 / 72.27  # ~7.09 inches


# ==============================================================================
# SAE Type Constants and Display Names
# ==============================================================================

# Display names for SAE types (maps config name prefixes to human-readable names)
# Includes both specific config names and generic type names for compatibility
SAE_TYPE_DISPLAY_NAMES = {
    'topk_k': 'TopKSAE',
    'archmsae_uw': 'ArchMSAE',
    'msae_rw': 'ActMSAE',
    # "hsae_361_16_6144_cc3m": "HSAE",
    "361_16_v2_cc3m": "HSAE-v2",
    "ewgsae": "EWGSAE",
    # 'archmsae_fw_k64_x8_cc3m_vit_20260209_105937': 'ArchMSAE_FW',
    # 'archmsae_uw_k64_x8_cc3m_vit_20260209_133945': 'ArchMSAE_AUX',
    # 'multisae': 'MultiSAE',
    'mpsae': 'MPSAE',
    # "unknown": "UnknownSAE",
}

# SAE type prefixes for extraction from config names
# Ordered by specificity (most specific first) to match correctly
SAE_TYPES = list(SAE_TYPE_DISPLAY_NAMES.keys())

# Colors for plotting SAE types
# Pick one palette by uncommenting one of the lines below.

# # Colors for plotting SAE types
# SAE_COLORS = {
#     'archmsae_uw_k64_x8_cc3m_vit_20260209_105919': '#2ca02c',
#     'archmsae_fw_k64_x8_cc3m_vit_20260209_105937': '#8FBC8F',
#     'archmsae_uw_k64_x8_cc3m_vit_20260209_133945': '#90EE90',
#     'topk_k': '#1f77b4',
#     'msae_rw': '#ff7f0e',
#     'multisae': '#d62728',
#     'mpsae': '#9467bd',
# }

SAE_COLORS_OKABE_ITO = {
    'archmsae_uw': '#009E73',
    'topk_k': '#0072B2',
    'msae_rw': '#D55E00',
    'hsae_361_16_6144_cc3m': '#CC79A7',
    '361_16_v2_cc3m': "#B679CC",
    'mpsae': '#E69F00',
    'ewgsae': '#56B4E9',
}

# SAE_COLORS = SAE_COLORS_OKABE_ITO
# SAE_COLORS = SAE_COLORS_DEEP
SAE_COLORS = SAE_COLORS_OKABE_ITO


# ==============================================================================
# Graph Creation Method Constants
# ==============================================================================

# Graph creation methods to extract from config names
GRAPH_METHODS = ['CONDACT', 'MCS', 'WONDACT']


# ==============================================================================
# Metric Constants
# ==============================================================================

# Standard metrics used across sweep and comparison analyses
METRICS = [
    # 'clarity_score',
    # 'child_greater_than_parent',
    'ms_score',
    'ms_child_greater_than_parent',
    'avg_intra_cluster_similarity',
    'conditional_activation',
    'fraction_parent_larger_than_child',
    'coverage',
    'conditional_spearman_corr_children_only_max_0',
]

# Extended metrics including num_children (used in sweep analyses)
METRICS_WITH_NUM_CHILDREN = [
    # 'clarity_score',
    # 'child_greater_than_parent',
    'num_children',
    'avg_intra_cluster_similarity',
    'conditional_activation',
    'fraction_parent_larger_than_child',
    'coverage',
    'conditional_spearman_corr_children_only_max_0',
    'ms_score',
    'ms_child_greater_than_parent',
]

# Display names for metrics (for visualization)
METRIC_DISPLAY_NAMES = {
    # 'clarity_score': 'Clarity',
    # 'child_greater_than_parent': 'HierarchicalAbstractness',
    'num_children': 'NumberOfChildren',
    'avg_intra_cluster_similarity': 'SiblingSimilarity',
    'conditional_activation': 'ActivationImplication',
    'fraction_parent_larger_than_child': 'ActivationDominance',
    'coverage': 'RefinementFrequency',
    'conditional_spearman_corr_children_only_max_0': 'ConditionalCorrelation',
    'ms_score': 'MonosemanticityScore',
    'ms_child_greater_than_parent': 'MSHierarchicalAbstractness',
}



# ==============================================================================
# Extraction Functions
# ==============================================================================

def extract_foundation_model(config_name):
    """
    Extract foundation model from config name.
    
    Args:
        config_name: Config simple name (e.g., "cc3m_vit_topk", "topk_...")
    
    Returns:
        String: 'vit', 'dinov2-base', or 'unknown'
    """
    config_lower = config_name.lower()
    if "_vit_" in config_lower:
        return "vit"
    elif "dino" in config_lower:
        return "dinov2"
    else:
        return "unknown"

def extract_sae_type(config_name):
    """
    Extract SAE type from config name.
    
    Matches config name against known SAE type prefixes, returning the
    longest matching prefix to handle specific variants (e.g., archmsae_uw)
    before more general types (e.g., archmsae).
    
    Args:
        config_name: Config simple name (e.g., "cc3m_vit_topk", "topk_...")
    
    Returns:
        String: SAE type matching one of the keys in SAE_TYPES, or 'unknown'
    """
    config_lower = config_name.lower()
    for sae_type in SAE_TYPES:
        if sae_type.lower() in config_lower:
            return sae_type
    return 'unknown'


def extract_graph_method(config_name):
    """
    Extract graph creation method from config name.
    
    Args:
        config_name: Config simple name
    
    Returns:
        String: 'CONDACT', 'MCS', or 'unknown'
    """
    config_upper = config_name.upper()
    for method in GRAPH_METHODS:
        if method in config_upper:
            return method
    return 'unknown'


def extract_threshold_from_graph_name(graph_name):
    """
    Extract threshold value from graph name.
    
    Args:
        graph_name: Graph configuration name (e.g., "dag_..._PAC0point95_...")
    
    Returns:
        Float threshold value or None if not found
    """
    # Look for PAC pattern (parent-given-child threshold for condact)
    pac_match = re.search(r'PAC(\d+point\d+)', graph_name)
    if pac_match:
        threshold_str = pac_match.group(1).replace('point', '.')
        return float(threshold_str)
    
    # Could add similar logic for MCS if needed
    return None


def extract_graph_type(graph_name):
    """
    Extract graph type from graph name.
    
    Args:
        graph_name: Graph configuration name
    
    Returns:
        String: 'condact', 'mcs', or 'unknown'
    """
    if "MCS" in graph_name:
        return "mcs"
    elif "CONDACT" in graph_name:
        return "condact"
    else:
        return "unknown"


def extract_mean_centering_mode(graph_name):
    # if _MC_ in graph_name then then MC else non-MC
    if "_MC_" in graph_name:
        return "MC"
    else:
        return "non-MC"


def summarize_config_name_components(config_names):
    """
    Summarize config-name-derived components used across reporting scripts.

    Args:
        config_names: Iterable of config name strings

    Returns:
        Dict with sorted unique values for sae_types, graph_methods, foundation_models
    """
    sae_types = sorted({extract_sae_type(config_name) for config_name in config_names})
    graph_methods = sorted({extract_graph_method(config_name) for config_name in config_names})
    foundation_models = sorted({extract_foundation_model(config_name) for config_name in config_names})

    return {
        'sae_types': sae_types,
        'graph_methods': graph_methods,
        'foundation_models': foundation_models,
    }


def append_config_summary_to_output_path(output_path, config_summary):
    """
    Append config-summary metadata to an output path stem.

    Args:
        output_path: Base output path string
        config_summary: Dict returned by summarize_config_name_components

    Returns:
        String path with metadata appended to stem
    """
    output_path = Path(output_path)
    sae_types_in_path = "_".join(config_summary['sae_types'])
    graph_methods_in_path = "_".join(config_summary['graph_methods'])
    foundation_models_in_path = "_".join(config_summary['foundation_models'])

    new_stem = (
        f"{output_path.stem}__saetypes_{sae_types_in_path}"
        f"__graphmethods_{graph_methods_in_path}"
        f"__foundmodels_{foundation_models_in_path}"
    )
    suffix = output_path.suffix
    filename = new_stem + suffix

    if len(filename.encode()) > 255:
        digest = hashlib.md5(filename.encode()).hexdigest()[:8]
        max_stem_bytes = 255 - len(suffix.encode()) - 9  # 8-char hash + underscore
        truncated = new_stem.encode()[:max_stem_bytes].decode("utf-8", errors="ignore")
        filename = f"{truncated}_{digest}{suffix}"

    return str(output_path.parent / filename)
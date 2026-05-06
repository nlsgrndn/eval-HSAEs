import gc
import numpy as np
import pandas as pd
from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterHierarchicalityData, ClusterData
from structure_evaluation.clustering_graph_metrics.Intra_cluster_metric_runner import MetricRunner
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
#from my_config import CONFIGS
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
from utils_sae_feature_properties import SAEDimensions
from structure_evaluation.clustering_graph_metrics.intra_hierarchy_hierarchicality_metrics import Metric, Scores
from structure_evaluation.clustering_graph_metrics.intra_hierarchy_hierarchicality_metrics import (
    ConsistencyWithHierarchilityScore
)

# ============================================================================
# MetricRunner - The Orchestrator
# ============================================================================

from typing import TypedDict

class HierarchicalityModelLevelData(TypedDict):
    interpretability_df: pd.DataFrame

class HierarchicalityMetricRunner(MetricRunner):
    """Orchestrates metric computation across configs and clusters for hierarchicality metrics."""

    FILENAME = "hierarchicality_metrics.pkl"
    
    def _get_model_level_data(self, config) -> pd.DataFrame:
        """Load interpretability data for a config."""
        interpretability_df = SAEDimensions(config=config).get_interpretability_data()
        return HierarchicalityModelLevelData(interpretability_df=interpretability_df)
    
    def _create_cluster_data(self, config_name: str, node: Any, parent_sae_id: int,
                            sae_ids_of_parent_subnodes: List[int], typed_dict: HierarchicalityModelLevelData) -> ClusterHierarchicalityData:
        """Extract and prepare cluster data."""
        # Get child SAE IDs (exclude parent)
        child_sae_ids = [sid for sid in sae_ids_of_parent_subnodes if sid != parent_sae_id]

        # Filter interpretability data for this cluster
        interpretability_df = typed_dict["interpretability_df"]
        interpretability_subset = interpretability_df[interpretability_df["sae_id"].isin(sae_ids_of_parent_subnodes+ [parent_sae_id])]
        
        return ClusterHierarchicalityData(
            config_name=config_name,
            node=node,
            parent_sae_id=parent_sae_id,
            child_sae_ids=child_sae_ids,
            interpretability_data=interpretability_subset
        )
    
if __name__ == "__main__":
    # Example usage
    metrics_to_run = [
        ConsistencyWithHierarchilityScore(column_name="clarity_score"),
        Scores(column_name="clarity_score"),
        ConsistencyWithHierarchilityScore(column_name="ms_score", prefix="ms_"),
        Scores(column_name="ms_score", prefix="ms_")
    ]
    
    config = DEFAULT_CONFIG
    runner = HierarchicalityMetricRunner(metrics=metrics_to_run)
    results_df = runner.compute_metrics_for_single_config(config.simple_name, config)
    HierarchicalityMetricRunner.save_results(results_df, config)

    

    # # Select configs
    # dataset_str = "cc3m"
    # model_str = "vit"
    # configs_dict = {key: value for key, value in CONFIGS.items() 
    #             if dataset_str in key and model_str in key}
    # # for debugging, limit to only the first 1 config
    # # configs_dict = dict(list(configs_dict.items())[:1])
    
    # # Loop over configs
    # all_dfs = []
    # runner = HierarchicalityMetricRunner(metrics=metrics_to_run)
    # for config_name, config in tqdm(configs_dict.items(), desc="Processing configs"):
    #     results_df = runner.compute_metrics_for_single_config(config_name, config)
    #     HierarchicalityMetricRunner.save_results(results_df, config)

    # for config_name, config in tqdm(configs_dict.items(), desc="Loading saved results"):
    #     results_df = HierarchicalityMetricRunner.load_results(config)
    #     all_dfs.append(results_df)
    

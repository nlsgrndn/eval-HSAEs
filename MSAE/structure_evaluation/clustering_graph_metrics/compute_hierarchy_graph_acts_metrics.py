import gc
import numpy as np
import pandas as pd
from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterActsData, IntraClusterActsMetric, ClusterData
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
# from my_config import CONFIGS
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
from utils_sae_feature_properties import SAEDimensions
from structure_evaluation.clustering_graph_metrics.intra_hierarchy_acts_metrics import (
    ConditionalCorrChildrenOnlyMetric,
    CoActivationCorrelationMetric,
    AverageChildrenActiveMetric,
    CoverageMetric,
    NumParentActiveMetric,
    ChildParentMagnitudeMetric,
    ChildActivationRatesMetric,
    MaskedCosineSimilarityWithParentMetric,
    ConditionalActivationMetric
    )
from activations_preprocessing.act_behav_utils import SAEActivationsPostProcessing, precompute_binary_activations, preprocess_continuous_activations
from configs.activation_preprocessing import get_acts_preprocess_cfg, ActivationsPreprocessingConfig
from structure_evaluation.clustering_graph_metrics.Intra_cluster_metric_runner import MetricRunner
from typing import TypedDict
from copy import deepcopy

NUM_DATAPOINTS = 500000



class ActsModelLevelData(TypedDict):
    activations: np.ndarray
    binarized_activations: np.ndarray

# ============================================================================
# MetricRunner - The Orchestrator
# ============================================================================

class ActsMetricRunner(MetricRunner):
    """Orchestrates metric computation across configs and clusters for activation-based metrics."""
    
    FILENAME = "acts_metrics.pkl"
    
    def __init__(self, metrics: List[IntraClusterActsMetric], activations_preprocessing_config: ActivationsPreprocessingConfig):
        super().__init__(metrics)

        self.activations_preprocessing_config = activations_preprocessing_config
        # if self.activations_preprocessing_config.max_num_samples is greater than NUM_DATAPOINTS, cap it. before create a copy to avoid modifying the original
        if self.activations_preprocessing_config.max_num_samples > NUM_DATAPOINTS:
            print(f"Capping max_num_samples from {self.activations_preprocessing_config.max_num_samples} to {NUM_DATAPOINTS}")
            self.activations_preprocessing_config = deepcopy(self.activations_preprocessing_config)
            self.activations_preprocessing_config.max_num_samples = NUM_DATAPOINTS

    
    def _get_model_level_data(self, config) -> ActsModelLevelData:
        """Load and preprocess activations for a config."""
        sae_activations_memmap = SAEDimensions(config=config).get_activations_memmap_of_graph_evaluation_dataset()
        activations = preprocess_continuous_activations(
            cond_act_metrics_config=self.activations_preprocessing_config,
            sae_activations_memmap=sae_activations_memmap,
            load_batch_size=10000
        )
        binary_activations = precompute_binary_activations(
            cond_act_metrics_config=self.activations_preprocessing_config,
            sae_activations_memmap=sae_activations_memmap,
            load_batch_size=10000,
        )

        return ActsModelLevelData(
            activations=activations,
            binarized_activations=binary_activations
        )
    
    def _create_cluster_data(self, config_name: str, node: Any, parent_sae_id: int,
                            sae_ids_of_parent_subnodes: List[int], typed_dict: ActsModelLevelData) -> ClusterActsData:
        """Extract and prepare cluster data."""
        activations = typed_dict["activations"]
        binary_activations = typed_dict["binarized_activations"]
        
        # Get child SAE IDs (exclude parent)
        child_sae_ids = [sid for sid in sae_ids_of_parent_subnodes if sid != parent_sae_id]
        
        # Get activations for parent and children
        parent_acts = activations[:, parent_sae_id]
        children_acts = activations[:, child_sae_ids]
        
        # Create binary masks
        parent_binary = binary_activations[:, parent_sae_id]
        children_binary = binary_activations[:, child_sae_ids]

        return ClusterActsData(
            config_name=config_name,
            node=node,
            parent_sae_id=parent_sae_id,
            child_sae_ids=child_sae_ids,
            parent_acts=parent_acts,
            children_acts=children_acts,
            parent_binary=parent_binary,
            children_binary=children_binary
        )
    

if __name__ == "__main__":
    # Example usage
    metrics_to_run = [
        CoActivationCorrelationMetric(),
        ChildParentMagnitudeMetric(),
        CoverageMetric(),
        AverageChildrenActiveMetric(),
        NumParentActiveMetric(),
        ChildActivationRatesMetric(),
        MaskedCosineSimilarityWithParentMetric(),
        ConditionalActivationMetric()
    ]
    cond_corr_thresholds = [0]
    aggregation_methods = ['max', 'sum']
    for threshold in cond_corr_thresholds:
        for method in aggregation_methods:
            metrics_to_run.append(ConditionalCorrChildrenOnlyMetric(threshold=threshold, aggregation_method=method))
    


    config = DEFAULT_CONFIG
    activations_preprocessing_config = get_acts_preprocess_cfg()
    assert activations_preprocessing_config.name in config.simple_name
    runner = ActsMetricRunner(metrics=metrics_to_run, activations_preprocessing_config=activations_preprocessing_config)
    results_df = runner.compute_metrics_for_single_config(config.simple_name, config)

    ActsMetricRunner.save_results(results_df, config)
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterHierarchicalityData, Metric

# # ============================================================================
# # Metric Base Class
# # ============================================================================

# class Metric(ABC):
#     """Base class for computing metrics on parent-child clusters."""
    
#     @abstractmethod
#     def compute(self, cluster_data: ClusterHierarchicalityData) -> Dict[str, Any]:
#         """
#         Compute metric and return results as a dict.
#         Values can be scalars, 1D arrays (per-child), or 2D arrays (matrices).

#         Args:
#             cluster_data (ClusterData): Data for the cluster to compute metrics on.
        
#         Returns:
#             Dict[str, Any]: Dictionary mapping metric names to their values.
#         """
#         pass



# ============================================================================
# Metric Implementations
# ============================================================================

class ConsistencyWithHierarchilityScore(Metric):
    """
    Computes consistency with hierarchicality based on interpretability scores.
    
    Compares parent and children clarity scores to assess if children are more specific
    (higher clarity) than their parent, as expected in a hierarchical structure.
    
    Returns:
    - child_greater_than_parent: binary indicator (1 if child > parent, 0 otherwise) (1D array)
    - child_parent_delta: difference (child - parent) (1D array)
    - child_parent_ratio: ratio (child / parent) (1D array)
    """
    
    def __init__(self, column_name: str, prefix: str = ""):
        self.column_name = column_name
        self.prefix = prefix

    def _validate_column(self, cluster_data: ClusterHierarchicalityData) -> None:
        if self.column_name not in cluster_data.interpretability_data.columns:
            raise KeyError(f"Missing required column '{self.column_name}' in interpretability_data")
    
    def compute(self, cluster_data: ClusterHierarchicalityData) -> Dict[str, Any]:
        self._validate_column(cluster_data)

        # Get parent clarity score
        parent_row = cluster_data.interpretability_data[
            cluster_data.interpretability_data['sae_id'] == cluster_data.parent_sae_id
        ]
        
        parent_score = parent_row[self.column_name].iloc[0]
        
        # Initialize arrays
        num_children = cluster_data.num_children
        child_greater_arr = np.full(num_children, np.nan)
        delta_arr = np.full(num_children, np.nan)
        ratio_arr = np.full(num_children, np.nan)
        
        # Compute metrics for each child
        for idx, child_sae_id in enumerate(cluster_data.child_sae_ids):
            child_row = cluster_data.interpretability_data[
                cluster_data.interpretability_data['sae_id'] == child_sae_id
            ]
            
            if len(child_row) > 0:
                child_score = child_row[self.column_name].iloc[0]
                
                # Binary: 1 if child > parent, 0 otherwise
                if np.isnan(child_score) or np.isnan(parent_score):
                    child_greater_arr[idx] = np.nan
                else:
                    child_greater_arr[idx] = float(child_score > parent_score)
                
                # Delta: child - parent
                delta_arr[idx] = child_score - parent_score
                
                # Ratio: child / parent (handle division by zero)
                if parent_score != 0:
                    ratio_arr[idx] = child_score / parent_score
                else:
                    ratio_arr[idx] = np.nan
        
        return {
            f'{self.prefix}child_greater_than_parent': child_greater_arr,
            f'{self.prefix}child_parent_delta': delta_arr,
            f'{self.prefix}child_parent_ratio': ratio_arr
        }

class Scores(Metric):
    """
    Returns raw scores for parent and children.
    
    Returns:
    - parent_score: score of parent (scalar)
    - child_scores: scores of children (1D array)
    """
    
    def __init__(self, column_name: str, prefix: str = ""):
        self.column_name = column_name
        self.prefix = prefix

    def _validate_column(self, cluster_data: ClusterHierarchicalityData) -> None:
        if self.column_name not in cluster_data.interpretability_data.columns:
            raise KeyError(f"Missing required column '{self.column_name}' in interpretability_data")
    
    def compute(self, cluster_data: ClusterHierarchicalityData) -> Dict[str, Any]:
        self._validate_column(cluster_data)

        # Get parent score
        parent_row = cluster_data.interpretability_data[
            cluster_data.interpretability_data['sae_id'] == cluster_data.parent_sae_id
        ]
        parent_score = parent_row[self.column_name].iloc[0]
        
        # Get child scores
        num_children = cluster_data.num_children
        child_scores_arr = np.full(num_children, np.nan)
        
        for idx, child_sae_id in enumerate(cluster_data.child_sae_ids):
            child_row = cluster_data.interpretability_data[
                cluster_data.interpretability_data['sae_id'] == child_sae_id
            ]
            
            child_scores_arr[idx] = child_row[self.column_name].iloc[0]
        
        return {
            f'{self.prefix}parent_score': parent_score,
            f'{self.prefix}child_scores': child_scores_arr
        }
from abc import ABC, abstractmethod
import numpy as np


from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class ClusterData:
    """Container for all cluster-related data needed for metric computation."""
    config_name: str
    node: Any
    parent_sae_id: int
    child_sae_ids: List[int]

    @property
    def num_children(self) -> int:
        return len(self.child_sae_ids)

@dataclass
class ClusterGeometryData(ClusterData):
    """Container for all cluster-related data needed for metric computation."""
    
    sim_matrix: np.ndarray  # shape: (num_children, num_children)

@dataclass
class ClusterSimilarityData(ClusterData):
    """Container for all cluster-related data needed for metric computation."""
    sim_matrix: np.ndarray  # shape: (num_children, num_children)
    baseline_mean: Dict[int, float]
    baseline_std: Dict[int, float]
    upper_bound_baseline_mean: Dict[int, float]
    upper_bound_baseline_std: Dict[int, float]
    upper_bound_baseline_max: Dict[int, float]

@dataclass
class IntravsInterFeatureSimilarityData(ClusterData):
    """Container for all cluster-related data needed for metric computation."""
    topk_embeddings: np.ndarray  # shape: (num_children, topk, embedding_dim)

@dataclass
class ClusterHierarchicalityData(ClusterData):
    """Container for all cluster-related data needed for metric computation."""
    
    interpretability_data: pd.DataFrame

@dataclass
class ClusterActsData(ClusterData):
    """Container for all cluster-related data needed for metric computation."""

    # Raw activations - shape (num_datapoints,) for parent, (num_datapoints, num_children) for children
    parent_acts: np.ndarray
    children_acts: np.ndarray

    # Binary masks - same shapes as activations
    parent_binary: np.ndarray
    children_binary: np.ndarray

    @property
    def num_datapoints(self) -> int:
        return len(self.parent_acts)


# ============================================================================
# Metric Base Class
# ============================================================================

class Metric(ABC):
    """Base class for computing metrics on parent-child clusters."""

    @abstractmethod
    def compute(self, cluster_data: ClusterData) -> Dict[str, Any]:
        """
        Compute metric and return results as a dict.
        Values can be scalars, 1D arrays (per-child), or 2D arrays (matrices).

        Args:
            cluster_data (ClusterData): Data for the cluster to compute metrics on.

        Returns:
            Dict[str, Any]: Dictionary mapping metric names to their values.
        """
        pass

class IntraClusterActsMetric(ABC):
    """Base class for computing metrics on parent-child clusters."""

    MIN_SAMPLES = 10  # Minimum activations for reliable statistics

    @abstractmethod
    def compute(self, cluster_data: ClusterActsData) -> Dict[str, Any]:
        """
        Compute metric and return results as a dict.
        Values can be scalars, 1D arrays (per-child), or 2D arrays (matrices).

        Args:
            cluster_data (ClusterData): Data for the cluster to compute metrics on.

        Returns:
            Dict[str, Any]: Dictionary mapping metric names to their values.
        """
        pass

    # def _filter_rare_children_from_matrix(self, matrix: np.ndarray,
    #                                       children_binary_active: np.ndarray) -> np.ndarray:
    #     """
    #     Set rows and columns to NaN for children with insufficient activation samples.

    #     Args:
    #         matrix: Square matrix of sibling metrics (num_children x num_children)
    #         children_binary_active: Binary mask of child activations when parent is active

    #     Returns:
    #         Filtered matrix with NaN for rare children's rows/columns
    #     """
    #     child_activation_counts = children_binary_active.sum(axis=0)
    #     num_children = len(child_activation_counts)

    #     for i in range(num_children):
    #         if child_activation_counts[i] < self.MIN_SAMPLES:
    #             matrix[i, :] = np.nan
    #             matrix[:, i] = np.nan

    #     return matrix





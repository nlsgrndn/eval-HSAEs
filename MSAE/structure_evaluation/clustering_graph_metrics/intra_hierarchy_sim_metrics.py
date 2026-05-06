import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterSimilarityData, Metric
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Metric Implementations
# ============================================================================

class IntraClusterEmbeddingSimilarity(Metric):
    """
    TODO: Description
    """
    
    def compute(self, cluster_data: ClusterSimilarityData) -> Dict[str, Any]:
        # check whether at least two children exist
        if cluster_data.num_children < 2:
            return {
                "delta_intra_cluster_similarity": np.nan,
                "upper_bound_mean_minus_lower_bound_mean": np.nan,
                "avg_intra_cluster_similarity": np.nan,
                "baseline_mean": cluster_data.baseline_mean,
                "baseline_std": cluster_data.baseline_std,
                "upper_bound_baseline_mean": cluster_data.upper_bound_baseline_mean,
                "upper_bound_baseline_std": cluster_data.upper_bound_baseline_std,
                "upper_bound_baseline_max": cluster_data.upper_bound_baseline_max
            }

        sim_matrix = cluster_data.sim_matrix  # shape: (num_children, num_children)
        upper_triangular_indices = np.triu_indices_from(sim_matrix, k=1)
        intra_similarities = sim_matrix[upper_triangular_indices]
        avg_intra_similarity = np.mean(intra_similarities)
        delta = avg_intra_similarity - cluster_data.baseline_mean
        upper_bound_mean_minus_lower_bound_mean = cluster_data.upper_bound_baseline_mean - cluster_data.baseline_mean
        upper_bound_baseline_mean = cluster_data.upper_bound_baseline_mean
        upper_bound_baseline_std = cluster_data.upper_bound_baseline_std
        upper_bound_baseline_max = cluster_data.upper_bound_baseline_max

        return {
            "delta_intra_cluster_similarity": delta,
            "upper_bound_mean_minus_lower_bound_mean": upper_bound_mean_minus_lower_bound_mean,
            "avg_intra_cluster_similarity": avg_intra_similarity,
            "baseline_mean": cluster_data.baseline_mean,
            "baseline_std": cluster_data.baseline_std,
            "upper_bound_baseline_mean": upper_bound_baseline_mean,
            "upper_bound_baseline_std": upper_bound_baseline_std,
            "upper_bound_baseline_max": upper_bound_baseline_max
        }

import numpy as np
import pandas as pd
from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterSimilarityData, Metric, ClusterData
from structure_evaluation.clustering_graph_metrics.Intra_cluster_metric_runner import MetricRunner
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
from sklearn.linear_model import LinearRegression
# from my_config import CONFIGS
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
from utils_sae_feature_properties import SAEDimensions
from structure_evaluation.clustering_graph_metrics.intra_hierarchy_sim_metrics import Metric, IntraClusterEmbeddingSimilarity
from structure_evaluation.structure_eval_utils import get_similarity_matrix
from typing import TypedDict
import os
import networkx as nx

from structure_extraction.structure_extraction_utils import get_graph

# ============================================================================
# MetricRunner - The Orchestrator
# ============================================================================

class SimilarityModelLevelData(TypedDict):
    sim_matrix: np.ndarray
    random_baseline_sims_mean: Dict[int, float]
    random_baseline_sims_std: Dict[int, float]
    topk_baseline_sims_mean: Dict[int, float]
    topk_baseline_sims_std: Dict[int, float]
    topk_baseline_max: Dict[int, float]

class SimilarityMetricRunner(MetricRunner):
    """Orchestrates metric computation across configs and clusters for similarity-based metrics."""

    FILENAME = "intra_hierarchy_similarity_metrics.pkl"
    SIM_DATA_KEY = "avg_embeddings"
    
    def __init__(self, metrics: List[Metric],):
        super().__init__(metrics)
    
    def _get_model_level_data(self, config) -> SimilarityModelLevelData:
        """Load similarity matrix for a config."""
        sim_matrix = get_similarity_matrix(self.SIM_DATA_KEY, config)
        G, _, _, _ = get_graph(config)
        baseline_means, baseline_stds = _calculate_baseline_scores_as_random_clusters_of_sizes_k(sim_matrix, G, num_trials=100)
        topk_baseline_means, topk_baseline_stds, topk_baseline_max = _calculate_upper_bound_baseline(sim_matrix, G, num_trials=100)

        return SimilarityModelLevelData(
            sim_matrix=sim_matrix,
            random_baseline_sims_mean=baseline_means,
            random_baseline_sims_std=baseline_stds,
            topk_baseline_sims_mean=topk_baseline_means,
            topk_baseline_sims_std=topk_baseline_stds,
            topk_baseline_max=topk_baseline_max
            )
    
    def _create_cluster_data(self, config_name: str, node: Any, parent_sae_id: int,
                            sae_ids_of_parent_subnodes: List[int], typed_dict: SimilarityModelLevelData) -> ClusterSimilarityData:
        """Extract and prepare cluster data."""
        
        # Get child SAE IDs (exclude parent)
        child_sae_ids = [sid for sid in sae_ids_of_parent_subnodes if sid != parent_sae_id]

        selected_sae_ids = child_sae_ids #TODO: check whether parent should be included
        sim_matrix = typed_dict["sim_matrix"][np.ix_(selected_sae_ids, selected_sae_ids)]
        baseline_mean = typed_dict["random_baseline_sims_mean"][len(selected_sae_ids)]
        baseline_std = typed_dict["random_baseline_sims_std"][len(selected_sae_ids)]
        topk_baseline_mean = typed_dict["topk_baseline_sims_mean"][len(selected_sae_ids)]
        topk_baseline_std = typed_dict["topk_baseline_sims_std"][len(selected_sae_ids)]
        topk_baseline_max = typed_dict["topk_baseline_max"][len(selected_sae_ids)]
        
        return ClusterSimilarityData(
            config_name=config_name,
            node=node,
            parent_sae_id=parent_sae_id,
            child_sae_ids=child_sae_ids,
            sim_matrix=sim_matrix,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            upper_bound_baseline_mean=topk_baseline_mean,
            upper_bound_baseline_std=topk_baseline_std,
            upper_bound_baseline_max=topk_baseline_max
        )



def _calculate_baseline_scores_as_random_clusters_of_sizes_k(similarity_matrix, G, num_trials=10):
    """
    Calculate baseline scores by randomly sampling clusters of the same sizes as actual clusters.
    
    Args:
        similarity_matrix: Pairwise similarity matrix for all SAE features
        num_trials: Number of random samples to generate for each cluster size (default: 100)
        
    Returns:
        tuple: (baseline_means, baseline_stds) where each is a dict mapping cluster size to mean/std baseline similarity
    """
    

    # Get all unique cluster sizes from the graph (excluding empty and full-graph clusters)
    # cluster_sizes = set([len(data.get("all_associated_original_sae_ids_in_subDAG", [])) for node_id, data in G.nodes(data=True)
    #                         if len(data.get("all_associated_original_sae_ids_in_subDAG", [])) > 0 and len(data.get("all_associated_original_sae_ids_in_subDAG", [])) != len(G.nodes())])
    
    cluster_sizes = set([len(list(G.successors(node_id))) for node_id in G.nodes()])


    # Get all available SAE IDs from the similarity matrix
    # get root node
    roots = [n for n, d in G.in_degree() if d == 0]
    assert len(roots) == 1, "Expected exactly one root node"
    root = roots[0]
    all_sae_ids_in_graph = G.nodes[root]["all_associated_original_sae_ids_in_subDAG"]
    #num_total_features = len(G.nodes[root]["all_associated_original_sae_ids_in_subDAG"])
    
    baseline_means = {}
    baseline_stds = {}
    
    for cluster_size in tqdm(cluster_sizes, desc="Calculating baseline scores for cluster sizes"):
        if cluster_size < 2:
            # Cannot compute pairwise similarity for cluster of size < 2
            baseline_means[cluster_size] = np.nan
            baseline_stds[cluster_size] = np.nan
            continue
            
        random_similarities = []
        
        for trial in range(num_trials):
            # Randomly sample N features without replacement
            random_sae_ids = np.random.choice(all_sae_ids_in_graph, size=cluster_size, replace=False)
            
            # Extract similarity submatrix for this random cluster
            similarity_submatrix = similarity_matrix[np.ix_(random_sae_ids, random_sae_ids)]
            
            # Calculate average pairwise similarity (upper triangular, excluding diagonal)
            upper_triangular_indices = np.triu_indices_from(similarity_submatrix, k=1)
            pairwise_similarities = similarity_submatrix[upper_triangular_indices]
            avg_similarity = np.mean(pairwise_similarities)
            
            random_similarities.append(avg_similarity)
        
        # Store the mean and std of random trials as the baseline for this cluster size
        baseline_means[cluster_size] = np.mean(random_similarities)
        baseline_stds[cluster_size] = np.std(random_similarities)
    
    return baseline_means, baseline_stds

def _calculate_upper_bound_baseline(similarity_matrix, G, num_trials=100):
    """
    Calculate upper bound baseline by selecting random seed features and their top-k most similar features.
    This creates maximally similar clusters to serve as an upper bound.
    
    Args:
        similarity_matrix: Pairwise similarity matrix for all SAE features
        G: Graph containing SAE features
        num_trials: Number of random samples to generate for each cluster size (default: 10)
        
    Returns:
        tuple: (baseline_means, baseline_stds) where each is a dict mapping cluster size to mean/std baseline similarity
    """
    
    print("POOR UPPER BOUND BASELINE CALCULATION METHOD; IDEALLY USE EMBEDDING BASED CLUSTERING INSTEAD")
    #import ipdb; ipdb.set_trace()
    # Problematic because 


    # # Get all unique cluster sizes from the graph (excluding empty and full-graph clusters)
    # cluster_sizes = set([len(data.get("all_associated_original_sae_ids_in_subDAG", [])) for node_id, data in G.nodes(data=True)
    #                         if len(data.get("all_associated_original_sae_ids_in_subDAG", [])) > 0 and len(data.get("all_associated_original_sae_ids_in_subDAG", [])) != len(G.nodes())])
    
    cluster_sizes = set([len(list(G.successors(node_id))) for node_id in G.nodes()])

    # Get all available SAE IDs from the similarity matrix
    # get root node
    roots = [n for n, d in G.in_degree() if d == 0]
    assert len(roots) == 1, "Expected exactly one root node"
    root = roots[0]
    all_sae_ids_in_graph = G.nodes[root]["all_associated_original_sae_ids_in_subDAG"]
    
    # Filter similarity matrix to only include features in the graph
    filtered_sim_matrix = similarity_matrix[np.ix_(all_sae_ids_in_graph, all_sae_ids_in_graph)]
    graph_feature_indices = np.arange(len(all_sae_ids_in_graph))
    
    baseline_max = {}
    baseline_means = {}
    baseline_stds = {}
    
    for cluster_size in tqdm(cluster_sizes, desc="Calculating upper bound baseline for cluster sizes"):
        if cluster_size < 2:
            # Cannot compute pairwise similarity for cluster of size < 2
            baseline_means[cluster_size] = np.nan
            baseline_stds[cluster_size] = np.nan
            baseline_max[cluster_size] = np.nan
            continue
            
        random_similarities = []
        
        for trial in range(num_trials):
            # Randomly select a seed feature from the graph
            seed_idx = np.random.choice(graph_feature_indices)
            
            # Get similarity scores for the seed feature
            sim_scores = filtered_sim_matrix[seed_idx, :]
            
            # Find top k-1 most similar features (excluding the seed itself)
            top_k_indices = np.argsort(sim_scores)[::-1]
            top_k_indices = top_k_indices[top_k_indices != seed_idx][:cluster_size-1]
            
            # Form cluster of seed + top k-1 similar features
            selected_indices = np.concatenate([[seed_idx], top_k_indices])
            
            # Extract similarity submatrix for this cluster
            similarity_submatrix = filtered_sim_matrix[np.ix_(selected_indices, selected_indices)]
            
            # Calculate average pairwise similarity (upper triangular, excluding diagonal)
            upper_triangular_indices = np.triu_indices_from(similarity_submatrix, k=1)
            pairwise_similarities = similarity_submatrix[upper_triangular_indices]
            avg_similarity = np.mean(pairwise_similarities)
            
            random_similarities.append(avg_similarity)
        
        # Store the mean and std of random trials as the baseline for this cluster size
        baseline_max[cluster_size] = np.max(random_similarities)
        baseline_means[cluster_size] = np.mean(random_similarities)
        baseline_stds[cluster_size] = np.std(random_similarities)
    
    return baseline_means, baseline_stds, baseline_max


if __name__ == "__main__":
    # Example usage
    metrics_to_run = [
        IntraClusterEmbeddingSimilarity(),
    ]



    config = DEFAULT_CONFIG
    runner = SimilarityMetricRunner(metrics=metrics_to_run)
    results_df = runner.compute_metrics_for_single_config(config.simple_name, config)
    SimilarityMetricRunner.save_results(results_df, config)
    
    # # Select configs
    # dataset_str = "cc3m"
    # model_str = "vit"
    # configs_dict = {key: value for key, value in CONFIGS.items() 
    #             if dataset_str in key and model_str in key}
    # # for debugging, limit to only the first 1 config
    # # configs_dict = dict(list(configs_dict.items())[:1])
    
    # # Loop over configs
    # all_dfs = []
    # runner = SimilarityMetricRunner(metrics=metrics_to_run)
    # for config_name, config in tqdm(configs_dict.items(), desc="Processing configs"):
    #     results_df = runner.compute_metrics_for_single_config(config_name, config)
    #     SimilarityMetricRunner.save_results(results_df, config)

    # for config_name, config in tqdm(configs_dict.items(), desc="Loading saved results"):
    #     results_df = SimilarityMetricRunner.load_results(config)
    #     all_dfs.append(results_df)
    
    # # Combine all results
    # results_df = pd.concat(all_dfs, ignore_index=True)
    
    # import ipdb; ipdb.set_trace()
    # # group by config name and get mean of each metric
    # selected_cols = [col for col in results_df.columns if col not in ['config_name', 'node_id', 'parent_sae_id', 'child_sae_ids']]
    # grouped_results = results_df.groupby("config_name")[selected_cols].mean()
    # print(grouped_results)

    # # save as pickle; use CONDITIONAL_GRAPH_NAME in filename
    # # save_path = f"intra_hierarchy_similarity_results_{CONDITIONAL_GRAPH_NAME}.pkl"
    # # results_df.to_pickle(save_path)

    # import ipdb; ipdb.set_trace()
    
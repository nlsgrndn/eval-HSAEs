"""
Script to aggregate hierarchy metrics for a single configuration.

This script loads metrics for a single config, computes summary statistics
for different metric types (single values, 1D arrays, 2D arrays), and saves
the aggregated results.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from structure_evaluation.clustering_graph_metrics.analysis_intra_hierarchy_metrics_helpers import (
    group_by,
    compute_single_value_metrics,
    compute_1d_array_metrics,
    compute_2d_array_metrics,
    load_metrics,
    load_node_metrics,
)

from my_config import DEFAULT_CONFIG
from structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_acts_metrics import ActsMetricRunner
from structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_depth_metrics import DepthMetricRunner
from structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_feature_level_metrics import FeatureLevelMetricRunner
from structure_evaluation.clustering_graph_metrics.compute_hierarchy_graph_metrics_based_on_feature_hierarchicality_metrics import HierarchicalityMetricRunner
from structure_evaluation.clustering_graph_metrics.compute_intra_hierarchy_graph_sim_metrics import SimilarityMetricRunner
from path_hub import PathBuilder


def detect_metric_types(df, exclude_columns=None):
    """
    Automatically detect metric types by inspecting the first non-null entry.
    
    Args:
        df: DataFrame with metrics
        exclude_columns: List of column names to exclude from detection.
                        If None, auto-detects metadata columns.
    
    Returns:
        Dictionary with keys 'single_value', 'array_1d', 'array_2d' containing lists of column names
    """
    if exclude_columns is None:
        # Define all possible metadata columns
        metadata_candidates = {
            # Cluster-level metadata
            'config_name', 'dataset_name', 'model_name',
            'node_id', 'parent_sae_id', 'child_sae_ids_x', 'child_sae_ids_y', 'child_sae_ids',
            # Node-level metadata
            'node', 'subset_sae_id', 'original_sae_id'
        }
        # Only exclude columns that actually exist in this DataFrame
        exclude_columns = [col for col in df.columns if col in metadata_candidates]
    
    metric_types = {
        'single_value': [],
        'array_1d': [],
        'array_2d': []
    }
    
    for col in df.columns:
        if col in exclude_columns:
            continue
        
        # Find first non-null value
        first_value = None
        for val in df[col]:
            if val is not None and (not isinstance(val, float) or not np.isnan(val)):
                first_value = val
                break
        
        if first_value is None:
            continue
        
        # Determine type
        if np.isscalar(first_value):
            metric_types['single_value'].append(col)
        elif isinstance(first_value, np.ndarray):
            if first_value.ndim == 1:
                metric_types['array_1d'].append(col)
            elif first_value.ndim == 2:
                metric_types['array_2d'].append(col)
        elif isinstance(first_value, (list, tuple)):
            # Also handle lists/tuples
            metric_types['array_1d'].append(col)
    
    return metric_types


def aggregate_all_metrics(df, metric_type="all"):
    """
    Aggregate metrics from the loaded DataFrame.
    
    Args:
        df: DataFrame loaded from parent_child_interaction_metrics.pkl
        metric_type: Which metrics to compute - "all", "single", "1d", or "2d"
    
    Returns:
        Dictionary with DataFrames for each metric type
    """
    # Automatically detect metric types
    detected_metrics = detect_metric_types(df)
    
    print(f"Detected metrics by type:")
    print(f"  Single value: {detected_metrics['single_value']}")
    print(f"  1D arrays: {detected_metrics['array_1d']}")
    print(f"  2D arrays: {detected_metrics['array_2d']}")
    
    # Group by configuration
    dfs_dict_by_config = group_by(df, "config_name")
    
    results = {}
    
    # Compute single value metrics
    if metric_type in ["all", "single"]:
        single_value_dfs = []
        for metric in detected_metrics['single_value']:
            single_value_dfs.append(compute_single_value_metrics(dfs_dict_by_config, metric))
        if single_value_dfs:
            results['single_value'] = pd.concat(single_value_dfs, ignore_index=True)
    
    # Compute 1D array metrics
    if metric_type in ["all", "1d"]:
        array_1d_dfs = []
        for metric in detected_metrics['array_1d']:
            array_1d_dfs.append(compute_1d_array_metrics(dfs_dict_by_config, metric))
        if array_1d_dfs:
            results['array_1d'] = pd.concat(array_1d_dfs, ignore_index=True)
    
    # Compute 2D array metrics
    if metric_type in ["all", "2d"]:
        array_2d_dfs = []
        for metric in detected_metrics['array_2d']:
            array_2d_dfs.append(compute_2d_array_metrics(dfs_dict_by_config, metric))
        if array_2d_dfs:
            results['array_2d'] = pd.concat(array_2d_dfs, ignore_index=True)
    
    return results


def aggregate_single_config(config, metrics_classes, metric_type="all"):
    """
    Aggregate metrics for a single configuration.
    
    Args:
        config: Configuration object
        metrics_classes: List of metric runner classes to compute
        metric_type: Which metrics to compute - "all", "single", "1d", or "2d"
    
    Returns:
        Aggregated metrics DataFrame
    """
    # Load metrics for single config
    metric_df = load_metrics([config], metrics_classes)
    
    # Check if we got any valid data
    if metric_df is None or metric_df.empty:
        #print(f"Error: No valid metric data found for config {config.simple_name}. Graph may be empty.")
        return pd.DataFrame()
    
    # Aggregate metrics
    results = aggregate_all_metrics(metric_df, metric_type=metric_type)
    
    # Combine all metric types into single DataFrame
    all_data = []
    for metric_type_name, result_df in results.items():
        all_data.append(result_df.copy())
    
    aggregated_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    return aggregated_df

def main(config, metric_type="all"):
    # Define metric classes by type
    cluster_metrics_classes = [ActsMetricRunner, HierarchicalityMetricRunner, SimilarityMetricRunner, DepthMetricRunner]
    node_level_metrics_classes = [FeatureLevelMetricRunner]
    
    all_aggregated_results = []
    
    # Aggregate cluster-level metrics
    if cluster_metrics_classes:
        print("\n=== Processing Cluster-Level Metrics ===")
        cluster_df = load_metrics([config], cluster_metrics_classes)
        
        if cluster_df is not None and not cluster_df.empty:
            cluster_results = aggregate_all_metrics(cluster_df, metric_type=metric_type)
            
            #  collect
            for metric_type_name, result_df in cluster_results.items():
                result_df = result_df.copy()
                all_aggregated_results.append(result_df)
        else:
            print("No cluster-level metrics found")
    
    # Aggregate node-level metrics
    if node_level_metrics_classes:
        print("\n=== Processing Node-Level Metrics ===")
        node_df = load_node_metrics([config], node_level_metrics_classes)
        
        if node_df is not None and not node_df.empty:
            node_results = aggregate_all_metrics(node_df, metric_type=metric_type)
            
            # collect
            for metric_type_name, result_df in node_results.items():
                result_df = result_df.copy()
                all_aggregated_results.append(result_df)
        else:
            print("No node-level metrics found")

    # Combine all results
    if all_aggregated_results:
        aggregated_df = pd.concat(all_aggregated_results, ignore_index=True)
    else:
        print("Warning: No metrics aggregated")
        aggregated_df = pd.DataFrame()

    return aggregated_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate hierarchy metrics for a single configuration"
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default="all",
        choices=["all", "single", "1d", "2d"],
        help="Which metric types to compute"
    )
    
    args = parser.parse_args()
    
    # Get config
    config = DEFAULT_CONFIG
    
    aggregated_df = main(config, metric_type=args.metric_type)



    # Save results using PathBuilder
    path_builder = PathBuilder(config=config)
    aggregated_path = path_builder.get_hierarchical_graph_aggregated_metrics_file_path()
    
    # Ensure parent directory exists
    Path(aggregated_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated metrics
    aggregated_df.to_pickle(aggregated_path)
    print(f"\nSaved aggregated metrics DataFrame to {aggregated_path}")
    print(f"Shape: {aggregated_df.shape}")


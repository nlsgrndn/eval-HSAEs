"""
Script to aggregate raw metric values for distribution plotting.

This script loads metrics for a single config and collects all raw values
(not just mean/std) for each metric, enabling distribution visualization.
Output format: DataFrame with columns [config_name, metric_name, values, n_values]
where 'values' is a list/array of all non-NaN values.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from structure_evaluation.clustering_graph_metrics.analysis_intra_hierarchy_metrics_helpers import (
    group_by,
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


def collect_single_value_raw(dfs_dict_by_config, metric_name):
    """
    Collect all raw values for a single-value metric.
    
    Args:
        dfs_dict_by_config: Dict of DataFrames grouped by config
        metric_name: Name of the metric
    
    Returns:
        DataFrame with columns [config_name, metric_name, values, n_values]
    """
    results = []
    for config_name, df in dfs_dict_by_config.items():
        values = df[metric_name].dropna().values
        results.append({
            'config_name': config_name,
            'metric_name': metric_name,
            'values': values.tolist(),
            'n_values': len(values)
        })
    return pd.DataFrame(results)


def collect_1d_array_raw(dfs_dict_by_config, metric_name):
    """
    Collect all raw values from 1D array metrics.
    
    Args:
        dfs_dict_by_config: Dict of DataFrames grouped by config
        metric_name: Name of the metric
    
    Returns:
        DataFrame with columns [config_name, metric_name, values, n_values]
    """
    results = []
    for config_name, df in dfs_dict_by_config.items():
        all_values = []
        for _, row in df.iterrows():
            arr = row[metric_name]
            if arr is not None:
                # Convert to numpy array if it's a list
                if isinstance(arr, list):
                    arr = np.array(arr)
                # Filter out NaNs
                valid_values = arr[~np.isnan(arr)]
                all_values.extend(valid_values)
        
        results.append({
            'config_name': config_name,
            'metric_name': metric_name,
            'values': all_values,
            'n_values': len(all_values)
        })
    return pd.DataFrame(results)


def collect_2d_array_raw(dfs_dict_by_config, metric_name):
    """
    Collect all raw values from 2D array metrics (upper triangle excluding diagonal).
    
    Args:
        dfs_dict_by_config: Dict of DataFrames grouped by config
        metric_name: Name of the metric
    
    Returns:
        DataFrame with columns [config_name, metric_name, values, n_values]
    """
    results = []
    for config_name, df in dfs_dict_by_config.items():
        all_values = []
        for matrix in df[metric_name]:
            if matrix is not None:
                # Get upper triangle excluding diagonal
                upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
                # Filter out NaNs
                valid_values = upper_triangle[~np.isnan(upper_triangle)]
                all_values.extend(valid_values)
        
        results.append({
            'config_name': config_name,
            'metric_name': metric_name,
            'values': all_values,
            'n_values': len(all_values)
        })
    return pd.DataFrame(results)


def collect_all_raw_values(df, metric_type="all"):
    """
    Collect raw values from all metrics in the DataFrame.
    
    Args:
        df: DataFrame loaded from metrics
        metric_type: Which metrics to collect - "all", "single", "1d", or "2d"
    
    Returns:
        DataFrame with columns [config_name, metric_name, values, n_values]
    """
    # Automatically detect metric types
    detected_metrics = detect_metric_types(df)
    
    print(f"Detected metrics by type:")
    print(f"  Single value: {detected_metrics['single_value']}")
    print(f"  1D arrays: {detected_metrics['array_1d']}")
    print(f"  2D arrays: {detected_metrics['array_2d']}")
    
    # Group by configuration
    dfs_dict_by_config = group_by(df, "config_name")
    
    all_results = []
    
    # Collect single value metrics
    if metric_type in ["all", "single"]:
        for metric in detected_metrics['single_value']:
            print(f"Collecting raw values for single-value metric: {metric}")
            result_df = collect_single_value_raw(dfs_dict_by_config, metric)
            all_results.append(result_df)
    
    # Collect 1D array metrics
    if metric_type in ["all", "1d"]:
        for metric in detected_metrics['array_1d']:
            print(f"Collecting raw values for 1D array metric: {metric}")
            result_df = collect_1d_array_raw(dfs_dict_by_config, metric)
            all_results.append(result_df)
    
    # Collect 2D array metrics
    if metric_type in ["all", "2d"]:
        for metric in detected_metrics['array_2d']:
            print(f"Collecting raw values for 2D array metric: {metric}")
            result_df = collect_2d_array_raw(dfs_dict_by_config, metric)
            all_results.append(result_df)
    
    # Combine all results
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def aggregate_single_config_raw_values(config, metrics_classes, metric_type="all"):
    """
    Collect raw metric values for a single configuration.
    
    Args:
        config: Configuration object
        metrics_classes: List of metric runner classes to compute
        metric_type: Which metrics to collect - "all", "single", "1d", or "2d"
    
    Returns:
        DataFrame with columns [config_name, metric_name, values, n_values]
    """
    # Load metrics for single config
    metric_df = load_metrics([config], metrics_classes)
    
    # Check if we got any valid data
    if metric_df is None or metric_df.empty:
        print(f"Warning: No valid metric data found for config {config.simple_name}")
        return pd.DataFrame()
    
    # Collect raw values
    raw_values_df = collect_all_raw_values(metric_df, metric_type=metric_type)
    
    return raw_values_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect raw metric values for distribution plotting"
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default="all",
        choices=["all", "single", "1d", "2d"],
        help="Which metric types to collect"
    )
    
    args = parser.parse_args()
    
    # Get config
    config = DEFAULT_CONFIG
    
    # Define metric classes by type
    cluster_metrics_classes = [ActsMetricRunner, HierarchicalityMetricRunner, SimilarityMetricRunner, DepthMetricRunner]
    node_level_metrics_classes = [FeatureLevelMetricRunner]
    
    all_raw_results = []
    
    print(f"\nCollecting raw values for config: {config.simple_name}")
    
    # Collect cluster-level raw values
    if cluster_metrics_classes:
        print("\n=== Collecting Cluster-Level Raw Values ===")
        cluster_df = load_metrics([config], cluster_metrics_classes)
        
        if cluster_df is not None and not cluster_df.empty:
            cluster_raw = collect_all_raw_values(cluster_df, metric_type=args.metric_type)
            if not cluster_raw.empty:
                all_raw_results.append(cluster_raw)
        else:
            print("No cluster-level metrics found")
    
    # Collect node-level raw values
    if node_level_metrics_classes:
        print("\n=== Collecting Node-Level Raw Values ===")
        node_df = load_node_metrics([config], node_level_metrics_classes)
        
        if node_df is not None and not node_df.empty:
            node_raw = collect_all_raw_values(node_df, metric_type=args.metric_type)
            if not node_raw.empty:
                all_raw_results.append(node_raw)
        else:
            print("No node-level metrics found")
    
    # Combine all results
    if all_raw_results:
        raw_values_df = pd.concat(all_raw_results, ignore_index=True)
    else:
        print("Warning: No raw values collected")
        raw_values_df = pd.DataFrame()

    # Save results using PathBuilder
    path_builder = PathBuilder(config=config)
    
    # Create output path
    raw_values_path = path_builder.get_hierarchical_graph_eval_path()
    raw_values_file = Path(raw_values_path) / "raw_values.pkl"
    
    # Ensure parent directory exists
    raw_values_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save raw values
    raw_values_df.to_pickle(raw_values_file)
    print(f"\nSaved raw values DataFrame to {raw_values_file}")
    print(f"Shape: {raw_values_df.shape}")
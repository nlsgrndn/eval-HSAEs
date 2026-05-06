"""
Script to create a comparison dataframe from multiple configurations.

This script loads aggregated metrics and metadata for multiple configs,
converts them to single-row format, and combines them into a comparison dataframe.
"""

import argparse
import pandas as pd
from pathlib import Path
from my_config import CONFIGS
from path_hub import PathBuilder
from configs.config_generator import (
    get_same_configs_with_mcs_graph,
    get_same_configs_with_default_graph_using_th1_activation_preprocessing, 
    get_same_configs_with_default_graph_using_nonMC_activation_preprocessing,
    get_same_configs_with_wondact_graph
)


def aggregated_to_single_row(aggregated_df, config_name):
    """
    Convert aggregated metrics DataFrame to single row format.
    
    Args:
        aggregated_df: DataFrame with columns [config_name, metric_name, mean, std, nan_proportion, ...]
        config_name: Name of the configuration
    
    Returns:
        Single-row DataFrame with columns config_name, metric1_mean, metric1_std, metric1_nan_proportion, ...
    """
    row_data = {'config_name': config_name}
    
    for _, row in aggregated_df.iterrows():
        metric_name = row['metric_name']
        row_data[f"{metric_name}_mean"] = row.get('mean', None)
        row_data[f"{metric_name}_std"] = row.get('std', None)
        row_data[f"{metric_name}_nan_proportion"] = row.get('nan_proportion', None)
    
    return pd.DataFrame([row_data])


def load_config_data(config):
    """
    Load aggregated metrics and metadata for a single config.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (aggregated_df, metadata_dict) or (None, None) if loading fails
    """
    path_builder = PathBuilder(config=config)
    
    # Load aggregated metrics
    try:
        aggregated_path = path_builder.get_hierarchical_graph_aggregated_metrics_file_path()
        aggregated_df = pd.read_pickle(aggregated_path)
        print(f"Loaded aggregated metrics from {aggregated_path}")
    except Exception as e:
        print(f"Error loading aggregated metrics for config {config.simple_name}: {e}")
        return None, None
    
    # Load metadata
    try:
        metadata_path = path_builder.get_hierarchical_graph_eval_metadata_file_path()
        metadata_df = pd.read_pickle(metadata_path)
        # Convert to dict for single config
        metadata_dict = metadata_df[metadata_df['config_name'] == config.simple_name].to_dict('records')[0] if not metadata_df.empty else {}
        metadata_dict.pop('config_name', None)  # Remove config_name as it will be added separately
        print(f"Loaded metadata from {metadata_path}")
    except Exception as e:
        print(f"Warning: Could not load metadata for config {config.simple_name}: {e}")
        metadata_dict = {}
    
    return aggregated_df, metadata_dict


def create_comparison_dataframe(configs, selected_metrics=None):
    """
    Create comparison dataframe from multiple configs.
    
    Args:
        configs: List of configuration objects
        selected_metrics: Optional list of metric names to include (if None, include all)
    
    Returns:
        DataFrame with one row per config
    """
    rows = []
    
    for config in configs:
        print(f"\nProcessing config: {config.simple_name}")
        
        # Load data
        aggregated_df, metadata_dict = load_config_data(config)
        
        if aggregated_df is None:
            print(f"Skipping config {config.simple_name} due to loading error")
            continue
        
        # Filter metrics if requested
        if selected_metrics:
            aggregated_df = aggregated_df[aggregated_df['metric_name'].isin(selected_metrics)]
        
        # Convert to single row
        single_row = aggregated_to_single_row(aggregated_df, config.simple_name)
        
        # Add metadata columns
        for key, value in metadata_dict.items():
            single_row[key] = value
        
        rows.append(single_row)
    
    # Combine all rows
    if not rows:
        print("Error: No valid data to combine")
        return None
    
    comparison_df = pd.concat(rows, ignore_index=True)
    return comparison_df


def get_configs(name):
    """Get all config objects."""
    if name == "condact_comparison":
        # assert that all config names contain CONDACT and _MC_
        for config in CONFIGS.values():
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" in config.simple_name, f"Config {config.simple_name} does not contain _MC_"
        return list(CONFIGS.values())
    elif name == "wondact_comparison_nonMC":
        # assert that all config names contain WONDACT and do not contain _MC_
        for config in CONFIGS.values():
            assert "WONDACT" in config.simple_name, f"Config {config.simple_name} does not contain WONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        return list(CONFIGS.values())
    elif name == "condact_comparison_nonMC":
        # assert that all config names contain CONDACT and do not contain _MC_
        for config in CONFIGS.values():
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        return list(CONFIGS.values())
    elif name == "condact_vs_mcs_comparison":
        condact_configs = list(CONFIGS.values())
        mcs_configs = get_same_configs_with_mcs_graph(list(CONFIGS.values()))
        for config in condact_configs:
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" in config.simple_name, f"Config {config.simple_name} does not contain _MC_"
        # assert that all config names contain MCS and _MC_
        for config in mcs_configs:
            assert "MCS" in config.simple_name, f"Config {config.simple_name} does not contain MCS"
            assert "_MC_" in config.simple_name, f"Config {config.simple_name} does not contain _MC_"
        configs = condact_configs + mcs_configs
        return configs
    elif name == "condact_vs_mcs_comparison_nonMC":
        condact_configs = list(CONFIGS.values())
        mcs_configs = get_same_configs_with_mcs_graph(list(CONFIGS.values()))
        for config in condact_configs:
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        for config in mcs_configs:
            assert "MCS" in config.simple_name, f"Config {config.simple_name} does not contain MCS"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        configs = condact_configs + mcs_configs
        return configs
    elif name == "condact_vs_wondact_comparison_nonMC":
        condact_configs = list(CONFIGS.values())
        wondact_configs = get_same_configs_with_wondact_graph(list(CONFIGS.values()))
        for config in condact_configs:
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        for config in wondact_configs:
            assert "WONDACT" in config.simple_name, f"Config {config.simple_name} does not contain WONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        configs = condact_configs + wondact_configs
        return configs
    elif name == "condact_MC_vs_nonMC_comparison":
        condact_configs = list(CONFIGS.values())
        nonMC_configs = get_same_configs_with_default_graph_using_nonMC_activation_preprocessing(list(CONFIGS.values()))
        for config in condact_configs:
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" in config.simple_name, f"Config {config.simple_name} does not contain _MC_"
        # assert that all config names contain CONDACT and do not contain _MC_
        for config in nonMC_configs:
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        configs = condact_configs + nonMC_configs
        return configs
    else:
        raise ValueError(f"Unknown comparison name: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create comparison dataframe from multiple configurations"
    )
    parser.add_argument(
        "--comparison-name",
        "-n",
        type=str,
        required=True,
        help="Name for this comparison"
    )
    # parser.add_argument(
    #     "--configs",
    #     type=str,
    #     nargs='+',
    #     required=True,
    #     help="Config names from CONFIGS to include"
    # )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='*',
        default=None,
        help="Optional: specific metrics to include (if not provided, all metrics are included)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments_results",
        help="Output directory for the comparison dataframe"
    )
    
    args = parser.parse_args()
    
    # # Get config objects
    # configs = []
    # for config_name in args.configs:
    #     if config_name not in CONFIGS:
    #         print(f"Warning: Config '{config_name}' not found in CONFIGS, skipping")
    #         continue
    #     configs.append(CONFIGS[config_name])
    
    # if not configs:
    #     print("Error: No valid configs found.")
    #     exit(1)

    configs = get_configs(args.comparison_name)
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(configs, selected_metrics=args.metrics)
    
    if comparison_df is None or comparison_df.empty:
        print("Error: Failed to create comparison dataframe")
        exit(1)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path_pkl = output_dir / f"{args.comparison_name}.pkl"
    output_path_csv = output_dir / f"{args.comparison_name}.csv"
    comparison_df.to_pickle(output_path_pkl)
    comparison_df.to_csv(output_path_csv, index=False)
    print(f"\nSaved comparison dataframe to {output_path_pkl}")
    print(f"Also saved CSV version to {output_path_csv}")
    print(f"Shape: {comparison_df.shape}")
    print(f"Configs: {len(comparison_df)}")

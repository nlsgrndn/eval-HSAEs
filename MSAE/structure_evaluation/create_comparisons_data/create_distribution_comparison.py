"""
Script to create a distribution comparison dataframe from multiple configurations.

This script loads raw metric values for multiple configs, adds metadata (SAE type,
graph method), and combines them into a single DataFrame for distribution visualization.
"""

import argparse
import pandas as pd
from pathlib import Path
from my_config import CONFIGS
from path_hub import PathBuilder
from configs.config_generator import (
    get_same_configs_with_mcs_graph,
    get_same_configs_with_default_graph_using_th1_activation_preprocessing, 
    get_same_configs_with_default_graph_using_nonMC_activation_preprocessing
)
from structure_evaluation.create_comparisons_data.comparison_utils import (
    extract_foundation_model,
    extract_sae_type,
    extract_graph_method,
    METRICS_WITH_NUM_CHILDREN
)


def load_single_config_raw_values(config):
    """
    Load raw_values.pkl for a single config.
    
    Args:
        config: Configuration object
    
    Returns:
        DataFrame with columns [config_name, metric_name, values, n_values]
        or None if loading fails
    """
    path_builder = PathBuilder(config=config)
    raw_values_path = Path(path_builder.get_hierarchical_graph_eval_path()) / "raw_values.pkl"
    
    try:
        df = pd.read_pickle(raw_values_path)
        print(f"Loaded raw values from {raw_values_path}")
        return df
    except Exception as e:
        print(f"Error loading raw values for config {config.simple_name}: {e}")
        return None


def create_distribution_comparison(configs, selected_metrics=None):
    """
    Combine raw values from multiple configs with metadata.
    
    Args:
        configs: List of configuration objects
        selected_metrics: Optional list of metric names to include (if None, include all)
    
    Returns:
        DataFrame with columns:
        [config_name, sae_type, graph_method, metric_name, values, n_values]
    """
    all_data = []
    
    for config in configs:
        print(f"\nProcessing config: {config.simple_name}")
        
        # Load raw values
        raw_df = load_single_config_raw_values(config)

        if raw_df is None or raw_df.empty:
            print(f"Skipping config {config.simple_name} due to loading error or empty data")
            continue
        
        # Filter metrics if requested
        if selected_metrics:
            raw_df = raw_df[raw_df['metric_name'].isin(selected_metrics)]
            if raw_df.empty:
                print(f"No matching metrics found for config {config.simple_name}")
                continue
        
        # Extract metadata
        sae_type = extract_sae_type(config.simple_name)
        graph_method = extract_graph_method(config.simple_name)
        
        # Add metadata columns
        raw_df['sae_type'] = sae_type
        raw_df['graph_method'] = graph_method
        
        print(f"Added {len(raw_df)} metric entries (sae_type={sae_type}, graph_method={graph_method})")
        
        all_data.append(raw_df)
    
    # Combine all
    if not all_data:
        print("Error: No valid data to combine")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def get_configs(name):
    """Get all config objects."""
    if name == "condact_comparison":
        # assert that all config names contain CONDACT and _MC_
        for config in CONFIGS.values():
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" in config.simple_name, f"Config {config.simple_name} does not contain _MC_"
        return list(CONFIGS.values())
    elif name == "mcs_comparison":
        configs = get_same_configs_with_mcs_graph(list(CONFIGS.values()))
        # assert that all config names contain MCS and _MC_
        for config in configs:
            assert "MCS" in config.simple_name, f"Config {config.simple_name} does not contain MCS"
            assert "_MC_" in config.simple_name, f"Config {config.simple_name} does not contain _MC_"
        return configs
    elif name == "condact_nonMC_comparison":
        configs = list(CONFIGS.values())
        configs = get_same_configs_with_default_graph_using_nonMC_activation_preprocessing(configs)
        # assert that all config names contain CONDACT and do not contain _MC_
        for config in configs:
            assert "CONDACT" in config.simple_name, f"Config {config.simple_name} does not contain CONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        return configs
    elif name == "wondact_nonMC_comparison":
        configs = list(CONFIGS.values())
        configs = get_same_configs_with_default_graph_using_nonMC_activation_preprocessing(configs)
        # assert that all config names contain CONDACT and do not contain _MC_
        for config in configs:
            assert "WONDACT" in config.simple_name, f"Config {config.simple_name} does not contain WONDACT"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        return configs
    elif name == "mcs_nonMC_comparison":
        configs = get_same_configs_with_mcs_graph(list(CONFIGS.values()))
        configs = get_same_configs_with_default_graph_using_nonMC_activation_preprocessing(configs)
        # assert that all config names contain MCS and do not contain _MC_
        for config in configs:
            assert "MCS" in config.simple_name, f"Config {config.simple_name} does not contain MCS"
            assert "_MC_" not in config.simple_name, f"Config {config.simple_name} contains _MC_"
        return configs
    else:
        raise ValueError(f"Unknown comparison name: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create distribution comparison dataframe from multiple configurations"
    )
    parser.add_argument(
        "--comparison-name",
        "-n",
        type=str,
        required=True,
        help="Name for this comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments_results",
        help="Output directory for the distribution comparison dataframe"
    )

    args = parser.parse_args()
    
    # Get all config objects
    configs = get_configs(args.comparison_name)
    
    # Verify graph methods
    graph_methods = set(extract_graph_method(config.simple_name) for config in configs)
    print(f"\nGraph methods in configs: {graph_methods}")
    
    # Verify SAE types
    sae_types = set(extract_sae_type(config.simple_name) for config in configs)
    print(f"SAE types in configs: {sae_types}")
    
    # Create distribution comparison dataframe
    print("\n" + "="*80)
    print("Creating distribution comparison dataframe...")
    print("="*80)
    
    distribution_df = create_distribution_comparison(configs, selected_metrics=METRICS_WITH_NUM_CHILDREN)
    
    if distribution_df is None or distribution_df.empty:
        print("Error: Failed to create distribution comparison dataframe")
        exit(1)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    foundation_models = distribution_df['config_name'].apply(extract_foundation_model).unique().tolist()
    assert len(foundation_models) == 1, "Expected exactly one foundation model in distribution comparison"
    output_path_pkl = output_dir / f"{args.comparison_name}_distributions.pkl"
    output_path_csv = output_dir / f"{args.comparison_name}_distributions_{foundation_models[0]}.csv"
    distribution_df.to_pickle(output_path_pkl)
    distribution_df.to_csv(output_path_csv, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Saved distribution comparison dataframe to {output_path_pkl}")
    print(f"Also saved CSV version to {output_path_csv}")
    print(f"Shape: {distribution_df.shape}")
    print(f"Configs: {distribution_df['config_name'].nunique()}")
    print(f"SAE types: {sorted(distribution_df['sae_type'].unique())}")
    print(f"Graph methods: {sorted(distribution_df['graph_method'].unique())}")
    print(f"Metrics: {sorted(distribution_df['metric_name'].unique())}")
    print(f"\nTotal values collected: {distribution_df['n_values'].sum()}")

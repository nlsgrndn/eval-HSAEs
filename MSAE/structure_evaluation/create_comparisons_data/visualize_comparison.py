"""
Script to visualize comparison dataframes.

This script loads a comparison dataframe and formats it for visualization.
"""

import argparse
import pandas as pd
from pathlib import Path


# Define metric categories
METADATA_COLUMNS = [
    "num_nodes_in_graph",
    "num_edges_in_graph",
    "num_clusters_in_graph",
    "num_clusters_above_children_threshold",
    'avg_feature_density',
    "avg_cluster_size",
    "avg_indegree",
    "max_depth_longest_path",
    "max_depth",
    "num_nodes_at_depth_0",
    "num_nodes_at_depth_1",
    "num_nodes_at_depth_2",
    "num_nodes_at_depth_3",
    "num_nodes_at_depth_4",
    "num_nodes_at_depth_5",
    "num_nodes_at_depth_6",
    "num_nodes_at_depth_0_longest_path",
    "num_nodes_at_depth_1_longest_path",
    "num_nodes_at_depth_2_longest_path",
    "num_nodes_at_depth_3_longest_path",
    "num_nodes_at_depth_4_longest_path",
    "num_nodes_at_depth_5_longest_path",
    "num_nodes_at_depth_6_longest_path",
]

GENERAL_PROPERTIES = [
    'num_children',
    'num_both_active',
    'num_parent_active',
    'child_base_rates',
    'child_rates_over_parent',
]

STRUCTURE_LEVEL_METRICS = [
    'child_greater_than_parent',
    'child_parent_delta',
    'child_parent_ratio',
    'parent_score',
    'child_scores',
    'baseline_mean',
    'avg_intra_cluster_similarity',
    'upper_bound_baseline_max'
]

ACTIVATION_LEVEL_METRICS = [
    "avg_num_children_active",
    "coverage",
    "avg_fraction_children_active",
    "fraction_parent_larger_than_child",
    "avg_child_parent_ratio",
    "conditional_activation",
    'conditional_spearman_corr_children_only_max_0',
    'conditional_spearman_corr_children_only_sum_0',
    'conditional_cosine_similarity_children_only_max_0',
    'conditional_cosine_similarity_children_only_sum_0',
]

MOST_IMPORTANT_METRICS = [
    'clarity_score',
    'child_greater_than_parent',
    'avg_intra_cluster_similarity',
    "coverage",
    "conditional_activation",
    "fraction_parent_larger_than_child",
    'conditional_spearman_corr_children_only_max_0',
]



def format_metrics_for_display(comparison_df, metric_names, precision=4):
    """
    Format metrics from comparison dataframe for display.
    
    Args:
        comparison_df: DataFrame with columns config_name, metric1_mean, metric1_std, metric1_nan_proportion, ...
        metric_names: List of metric names (without _mean, _std, _nan_proportion suffixes)
        precision: Number of decimal places for formatting
    
    Returns:
        DataFrame with formatted metrics (mean ± std [nan_prop])
    """
    display_df = comparison_df[['config_name']].copy()
    
    for metric in metric_names:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        nan_col = f"{metric}_nan_proportion"
        
        # Check if columns exist
        if mean_col not in comparison_df.columns:
            print(f"Warning: Metric '{metric}' not found in dataframe")
            continue
        
        # Format mean ± std
        formatted = comparison_df[mean_col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A")
        
        if std_col in comparison_df.columns:
            formatted = formatted + " ± " + comparison_df[std_col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A")
        
        # Add nan_proportion in brackets if non-zero
        if nan_col in comparison_df.columns:
            nan_str = comparison_df[nan_col].apply(
                lambda x: f" [{x:.2f}]" if pd.notna(x) and x > 0 else ""
            )
            formatted = formatted + nan_str
        
        display_df[metric] = formatted
    
    return display_df


def format_metadata_for_display(comparison_df, metadata_columns, precision=2):
    """
    Format metadata columns for display.
    
    Args:
        comparison_df: DataFrame with metadata columns
        metadata_columns: List of metadata column names
        precision: Number of decimal places for formatting
    
    Returns:
        DataFrame with formatted metadata
    """
    display_df = comparison_df[['config_name']].copy()
    
    for col in metadata_columns:
        if col not in comparison_df.columns:
            print(f"Warning: Metadata column '{col}' not found in dataframe")
            continue
        
        # Format based on type
        if comparison_df[col].dtype in ['int64', 'int32']:
            display_df[col] = comparison_df[col].astype(str)
        else:
            display_df[col] = comparison_df[col].apply(
                lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A"
            )
    
    return display_df

def create_structural_metrics_table(comparison_df, precision=4):
    """
    Create a table combining metadata and num_children metric.
    
    Args:
        comparison_df: DataFrame with metadata columns and metrics
        precision: Number of decimal places for num_children metric
    
    Returns:
        DataFrame with formatted structural information
    """
    display_df = comparison_df[['config_name']].copy()
    
    # Add metadata columns
    for col in METADATA_COLUMNS:
        if col not in comparison_df.columns:
            print(f"Warning: Metadata column '{col}' not found in dataframe")
            continue
        
        # Format based on type
        if comparison_df[col].dtype in ['int64', 'int32']:
            display_df[col] = comparison_df[col].astype(str)
        else:
            display_df[col] = comparison_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )
    
    # Add num_children metric (formatted as mean ± std)
    metric = 'num_children'
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    nan_col = f"{metric}_nan_proportion"
    
    if mean_col in comparison_df.columns:
        # Format mean ± std
        formatted = comparison_df[mean_col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A")
        
        if std_col in comparison_df.columns:
            formatted = formatted + " ± " + comparison_df[std_col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A")
        
        # Add nan_proportion in brackets if non-zero
        if nan_col in comparison_df.columns:
            nan_str = comparison_df[nan_col].apply(
                lambda x: f" [{x:.2f}]" if pd.notna(x) and x > 0 else ""
            )
            formatted = formatted + nan_str
        
        display_df[metric] = formatted
    else:
        print(f"Warning: Metric 'num_children' not found in dataframe")
    
    return display_df


def visualize_table(display_df, title="Metrics"):
    """
    Print formatted table to console.
    
    Args:
        display_df: DataFrame to display
        title: Title for the table
    """
    print(f"\n{title}")
    print("=" * 120)
    print(display_df.to_string(index=False))
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize comparison dataframe')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to comparison dataframe pickle file'
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=4,
        help='Number of decimal places for metrics (default: 4)'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export formatted tables as pickle files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments_results',
        help='Output directory for exported tables (default: experiments_results)'
    )
    
    args = parser.parse_args()
    
    # Load comparison dataframe
    print(f"Loading comparison dataframe from {args.input}...")
    comparison_df = pd.read_pickle(args.input)
    
    # Truncate config names if requested
    comparison_df['config_name'] = comparison_df['config_name']
    
    print(f"Loaded {len(comparison_df)} configurations")
    print(f"Columns: {len(comparison_df.columns)}")
    
    # Format and display metrics
    print("\n" + "="*120)
    print("STRUCTURAL METRICS")
    print("="*120)
    structural_metrics_display = create_structural_metrics_table(
        comparison_df, precision=args.precision
    )
    visualize_table(structural_metrics_display, title="Structural Metrics (Graph Statistics + Num Children)")
    
    # print("\n" + "="*120)
    # print("MOST IMPORTANT METRICS")
    # print("="*120)
    # most_important_display = format_metrics_for_display(
    #     comparison_df, MOST_IMPORTANT_METRICS, precision=args.precision
    # )
    # visualize_table(most_important_display, title="Most Important Metrics")
    
    # print("\n" + "="*120)
    # print("STRUCTURE-LEVEL METRICS")
    # print("="*120)
    # structure_display = format_metrics_for_display(
    #     comparison_df, STRUCTURE_LEVEL_METRICS, precision=args.precision
    # )
    # visualize_table(structure_display, title="Structure-Level Metrics")
    
    # print("\n" + "="*120)
    # print("GENERAL PROPERTIES")
    # print("="*120)
    # general_display = format_metrics_for_display(
    #     comparison_df, GENERAL_PROPERTIES, precision=args.precision
    # )
    # visualize_table(general_display, title="General Properties")
    
    # print("\n" + "="*120)
    # print("ACTIVATION-LEVEL METRICS")
    # print("="*120)
    # activation_display = format_metrics_for_display(
    #     comparison_df, ACTIVATION_LEVEL_METRICS, precision=args.precision
    # )
    # visualize_table(activation_display, title="Activation-Level Metrics (Activation Desiderata)")
    
    # print("\n" + "="*120)
    # print("METADATA")
    # print("="*120)
    # metadata_display = format_metadata_for_display(
    #     comparison_df, METADATA_COLUMNS, precision=2
    # )
    # visualize_table(metadata_display, title="Metadata")
    
    # # Display NaN proportions summary
    # print("\n" + "="*120)
    # print("NaN PROPORTIONS SUMMARY")
    # print("="*120)
    # all_metrics = STRUCTURE_LEVEL_METRICS + GENERAL_PROPERTIES + ACTIVATION_LEVEL_METRICS
    # nan_cols = [f"{m}_nan_proportion" for m in all_metrics if f"{m}_nan_proportion" in comparison_df.columns]
    
    # if nan_cols:
    #     nan_df = comparison_df[['config_name'] + nan_cols].copy()
    #     # Rename columns to remove _nan_proportion suffix
    #     nan_df.columns = ['config_name'] + [col.replace('_nan_proportion', '') for col in nan_cols]
    #     # Only show columns with non-zero values
    #     non_zero_cols = ['config_name']
    #     for col in nan_df.columns[1:]:
    #         if (nan_df[col] > 0).any():
    #             non_zero_cols.append(col)
        
    #     if len(non_zero_cols) > 1:
    #         print(nan_df[non_zero_cols].to_string(index=False))
    #     else:
    #         print("No metrics with NaN values")
    # print("\n")
    
    # Export if requested
    if args.export:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract comparison name from input filename
        comparison_name = Path(args.input).stem
        
        print(f"\nExporting formatted tables to {output_dir}/...")
        
        structural_metrics_display.to_pickle(output_dir / f"{comparison_name}_structural_metrics.pkl")
        # most_important_display.to_pickle(output_dir / f"{comparison_name}_most_important.pkl")
        # structure_display.to_pickle(output_dir / f"{comparison_name}_structure.pkl")
        # general_display.to_pickle(output_dir / f"{comparison_name}_general.pkl")
        # activation_display.to_pickle(output_dir / f"{comparison_name}_activation.pkl")
        # metadata_display.to_pickle(output_dir / f"{comparison_name}_metadata.pkl")
        
        print(f"Exported tables successfully to {output_dir}/")


if __name__ == "__main__":
    main()

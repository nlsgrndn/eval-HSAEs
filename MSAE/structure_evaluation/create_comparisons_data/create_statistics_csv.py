"""
Script to create LaTeX tables from structural metrics table.

This script loads a structural metrics table and converts it to LaTeX format,
using display names from comparison_utils for consistent naming.
"""

import argparse
import pandas as pd
from pathlib import Path
from structure_evaluation.create_comparisons_data.comparison_utils import (
    SAE_TYPE_DISPLAY_NAMES,
    extract_sae_type,
    extract_graph_method,
    summarize_config_name_components,
    append_config_summary_to_output_path,
)





def simplify_config_name(config_name, extract_type=True, extract_method=True):
    """
    Simplify config name by extracting SAE type and graph method.
    
    Args:
        config_name: Full config name
        extract_type: Whether to extract SAE type display name
        extract_method: Whether to extract graph method
    
    Returns:
        Simplified name string
    """
    parts = []
    
    if extract_type:
        sae_type = extract_sae_type(config_name)
        display_name = SAE_TYPE_DISPLAY_NAMES.get(sae_type, sae_type)
        parts.append(display_name)
    
    if extract_method:
        graph_method = extract_graph_method(config_name)
        if graph_method != 'unknown':
            parts.append(graph_method)
    
    if parts:
        return " - ".join(parts)
    else:
        return config_name


def create_csv(structural_df, simplify_names=True,):
    df = structural_df.copy()
    
    # Simplify config names if requested
    if simplify_names:
        df['config_name'] = df['config_name'].apply(
            lambda x: simplify_config_name(x, extract_type=True, extract_method=True)
        )
    
    # Rename config_name column
    df = df.rename(columns={'config_name': 'Configuration'})
    
    # Rename other columns to more readable names
    column_renames = {
        'num_nodes_in_graph': 'Nodes',
        'num_edges_in_graph': 'Edges',
        'num_clusters_in_graph': 'Clusters',
        'num_clusters_above_children_threshold': 'Clusters (>Thresh)',
        'num_children': 'Num Children'
    }
    df = df.rename(columns=column_renames)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create LaTeX table from structural metrics table'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to structural metrics pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save LaTeX output (if not specified, prints to stdout)'
    )
    args = parser.parse_args()
    
    # Load structural metrics table
    print(f"Loading structural metrics from {args.input}...")
    structural_df = pd.read_pickle(args.input)
    
    print(f"Loaded {len(structural_df)} configurations")
    print(f"Columns: {list(structural_df.columns)}")

    config_summary = summarize_config_name_components(structural_df['config_name'])

    print(f"\nSAE types found: {config_summary['sae_types']}")
    print(f"Graph methods found: {config_summary['graph_methods']}")
    print(f"Foundation models found: {config_summary['foundation_models']}")

    # Add sae_types, graph_methods, and foundation_models to output path for organization
    if args.output:
        args.output = append_config_summary_to_output_path(args.output, config_summary)
        print(f"Output path updated to include config info: {args.output}")
    

    structural_df = create_csv(structural_df, simplify_names=True)

    # save as csv
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        structural_df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"Structural metrics saved to {output_path.with_suffix('.csv')}")

if __name__ == "__main__":
    main()

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


def parse_formatted_metric(metric_str):
    """
    Parse a formatted metric string (e.g., "1.2345 ± 0.0123 [0.05]") into components.
    
    Args:
        metric_str: Formatted metric string
    
    Returns:
        Tuple of (mean_str, std_str, nan_prop_str) or None if parsing fails
    """
    if pd.isna(metric_str) or metric_str == "N/A":
        return None
    
    # Split by ±
    parts = metric_str.split(" ± ")
    if len(parts) == 2:
        mean_str = parts[0].strip()
        rest = parts[1].strip()
        
        # Check for nan proportion in brackets
        if "[" in rest:
            std_str = rest.split("[")[0].strip()
            nan_prop_str = rest.split("[")[1].replace("]", "").strip()
            return (mean_str, std_str, nan_prop_str)
        else:
            return (mean_str, rest, None)
    elif len(parts) == 1:
        # Just a mean value, no std
        mean_str = parts[0].strip()
        if "[" in mean_str:
            mean_only = mean_str.split("[")[0].strip()
            nan_prop_str = mean_str.split("[")[1].replace("]", "").strip()
            return (mean_only, None, nan_prop_str)
        else:
            return (mean_str, None, None)
    
    return None


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


def format_number_for_latex(value_str, precision=None):
    """
    Format a number string for LaTeX (handle scientific notation, etc).
    
    Args:
        value_str: String representation of number
        precision: Optional precision to round to
    
    Returns:
        LaTeX-formatted string
    """
    if value_str is None or value_str == "N/A":
        return "N/A"
    
    try:
        val = float(value_str)
        if precision is not None:
            val_str = f"{val:.{precision}f}"
        else:
            val_str = value_str
        
        # Handle scientific notation if needed
        if "e" in val_str.lower():
            # Convert to LaTeX scientific notation
            base, exp = val_str.lower().split("e")
            return f"{base} \\times 10^{{{int(exp)}}}"
        else:
            return val_str
    except (ValueError, AttributeError):
        return value_str


def create_latex_table(structural_df, simplify_names=True, show_std=True, 
                       caption="Structural Metrics", label="tab:structural_metrics"):
    """
    Create a LaTeX table from structural metrics dataframe.
    
    Args:
        structural_df: DataFrame with structural metrics
        simplify_names: Whether to simplify config names
        show_std: Whether to show standard deviation in num_children
        caption: Table caption
        label: Table label
    
    Returns:
        String containing LaTeX table code
    """
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
    
    # Process num_children column if present and std should be shown
    if 'Num Children' in df.columns and not show_std:
        # Extract just the mean value (before ±)
        df['Num Children'] = df['Num Children'].apply(
            lambda x: x.split(" ± ")[0].strip() if isinstance(x, str) and " ± " in x else x
        )
    
    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    
    # Determine column alignment
    n_cols = len(df.columns)
    col_alignment = "l" + "r" * (n_cols - 1)  # Left-align first column, right-align others
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_alignment}}}")
    latex_lines.append("\\toprule")
    
    # Header row
    header = " & ".join(df.columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        row_values = [str(val) for val in row.values]
        row_str = " & ".join(row_values) + " \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


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
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Do not simplify config names'
    )
    parser.add_argument(
        '--no-std',
        action='store_true',
        help='Do not show standard deviation for num_children'
    )
    parser.add_argument(
        '--caption',
        type=str,
        default='Structural Metrics',
        help='Table caption (default: "Structural Metrics")'
    )
    parser.add_argument(
        '--label',
        type=str,
        default='tab:structural_metrics',
        help='Table label (default: "tab:structural_metrics")'
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
    
    # Create LaTeX table
    print("\nCreating LaTeX table...")

    latex_table = create_latex_table(
        structural_df,
        simplify_names=not args.no_simplify,
        show_std=not args.no_std,
        caption=args.caption,
        label=args.label
    )
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {args.output}")
    else:
        print("\n" + "="*80)
        print("LATEX TABLE")
        print("="*80)
        print(latex_table)
        print("="*80)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

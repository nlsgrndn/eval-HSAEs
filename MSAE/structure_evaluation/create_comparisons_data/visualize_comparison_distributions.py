"""
Script to visualize distribution comparison dataframes as violin plots.

This script loads a distribution comparison dataframe and creates violin plot
visualizations showing the full distribution of metrics across different SAE types.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.stats import truncnorm
from pathlib import Path
from structure_evaluation.create_comparisons_data.comparison_utils import (
    SAE_TYPE_DISPLAY_NAMES,
    SAE_TYPES,
    SAE_COLORS,
    METRIC_DISPLAY_NAMES,
    METRICS,
    TEXTWIDTH_INCHES,
    summarize_config_name_components,
    append_config_summary_to_output_path,
)

N_ROWS, N_COLS = 4, 2
SUBPLOT_HEIGHT = 1.8  # inches per row
COMBINED_FIGSIZE = (TEXTWIDTH_INCHES, SUBPLOT_HEIGHT * N_ROWS)

BASE_FONT_SIZE = 7
plt.rcParams.update({
    'font.size':        BASE_FONT_SIZE,
    'axes.titlesize':   BASE_FONT_SIZE + 1,
    'axes.labelsize':   BASE_FONT_SIZE,
    'xtick.labelsize':  BASE_FONT_SIZE - 1,
    'ytick.labelsize':  BASE_FONT_SIZE - 1,
    'legend.fontsize':  BASE_FONT_SIZE - 1,
    'lines.linewidth':  1.0,
    'lines.markersize': 3,
})


def prepare_long_format(distributions_df, selected_metrics=None):
    """
    Transform distribution data from storage format to long plotting format.
    
    Args:
        distributions_df: DataFrame with columns [config_name, sae_type, graph_method, 
                         metric_name, values, n_values]
        selected_metrics: Optional list of metrics to include (if None, use all)
    
    Returns:
        DataFrame in long format with columns [sae_type, metric_name, value]
        where each row represents a single value
    """
    # Filter metrics if specified
    if selected_metrics:
        distributions_df = distributions_df[distributions_df['metric_name'].isin(selected_metrics)]
    
    # Expand 'values' lists into individual rows
    rows = []
    for _, row in distributions_df.iterrows():
        sae_type = row['sae_type']
        metric_name = row['metric_name']
        
        # Expand list of values into separate rows
        for val in row['values']:
            rows.append({
                'sae_type': sae_type,
                'metric_name': metric_name,
                'value': val
            })
    
    return pd.DataFrame(rows)


def create_violin_plot_for_metric(metric_data, metric_name, figsize=(TEXTWIDTH_INCHES, TEXTWIDTH_INCHES / 1.618), filename=None, ax=None):
    """
    Create a single violin plot for one metric.
    
    Args:
        metric_data: Long-format DataFrame with columns [sae_type, value] for one metric
        metric_name: Name of the metric
        figsize: Figure size (width, height)
        filename: Optional filename to display as subtitle
        ax: Optional matplotlib axis to plot on (if None, creates new figure)
    
    Returns:
        Axis object
    """
    # Apply display names to SAE types
    metric_data = metric_data.copy()
    metric_data['sae_display'] = metric_data['sae_type'].map(
        lambda x: SAE_TYPE_DISPLAY_NAMES.get(x, x)
    )
    
    # Canonical ordering and colors from SAE_TYPES
    available_keys = set(metric_data['sae_type'].unique())
    ordered_keys = [st for st in SAE_TYPES if st in available_keys]
    ordered_keys += sorted(available_keys - set(ordered_keys))
    order = [SAE_TYPE_DISPLAY_NAMES.get(k, k) for k in ordered_keys]
    palette = {SAE_TYPE_DISPLAY_NAMES.get(k, k): SAE_COLORS.get(k, '#333333') for k in ordered_keys}

    # Create figure if no axis provided
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    sns.violinplot(
        data=metric_data,
        x='sae_display',
        y='value',
        order=order,
        inner='box',  # Show box plot inside violin
        ax=ax,
        palette=palette,
        cut=0  # Don't extend violin past observed data
    )
    
    # Get display name for metric
    metric_display = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    # Formatting
    ax.set_ylabel('')
    
    # Set title (with optional subtitle for filename)
    if filename:
        ax.set_title(f'{metric_display}\n{filename}', pad=4)
    else:
        ax.set_title(metric_display, pad=4)
    
    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    if metric_name != 'num_children':
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    if created_fig:
        plt.tight_layout()
    
    return ax


def create_raincloud_plot_for_metric(metric_data, metric_name, figsize=(TEXTWIDTH_INCHES, TEXTWIDTH_INCHES / 1.618), max_points=7000, filename=None, ax=None):
    """
    Create a raincloud plot for one metric showing density, box plot, and individual points.
    
    A raincloud plot combines:
    - Half-violin (density on one side)
    - Box plot (quartiles in the middle)
    - Strip plot (individual jittered points on the other side)
    
    Args:
        metric_data: Long-format DataFrame with columns [sae_type, value] for one metric
        metric_name: Name of the metric
        figsize: Figure size (width, height)
        max_points: Maximum points to show per SAE type (downsampled if exceeded)
        filename: Optional filename to display as subtitle
        ax: Optional matplotlib axis to plot on (if None, creates new figure)
    
    Returns:
        Axis object
    """
    # Apply display names to SAE types
    metric_data = metric_data.copy()
    metric_data['sae_display'] = metric_data['sae_type'].map(
        lambda x: SAE_TYPE_DISPLAY_NAMES.get(x, x)
    )
    
    # Create figure if no axis provided
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Canonical ordering and colors from SAE_TYPES
    available_keys = set(metric_data['sae_type'].unique())
    ordered_keys = [st for st in SAE_TYPES if st in available_keys]
    ordered_keys += sorted(available_keys - set(ordered_keys))
    sae_types = [SAE_TYPE_DISPLAY_NAMES.get(k, k) for k in ordered_keys]
    colors = [SAE_COLORS.get(k, '#333333') for k in ordered_keys]


    LEFT_OFFSET = -0.15
    RIGHT_OFFSET = 0.2
    JITTER_STD = 0.04
    jitter_low = -0.1 / JITTER_STD   # lower bound: i + RIGHT_OFFSET - 0.1 (just past box right edge)
    jitter_up = 0.1 / JITTER_STD     # upper bound: i + RIGHT_OFFSET + 0.1

    # 1. Half-violin (density) on the left side
    # Use split=True style by manually offsetting
    violin_parts = ax.violinplot(
        [metric_data[metric_data['sae_display'] == sae]['value'].values for sae in sae_types],
        positions=np.arange(len(sae_types)) + LEFT_OFFSET,  # Shift left for half-violin effect
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violins and shift them to the left
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        # Get the vertices and shift left by modifying x-coordinates
        vertices = pc.get_paths()[0].vertices
        vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, np.arange(len(sae_types))[i] + LEFT_OFFSET)
    
    # 2. Box plot (quartiles) in the middle-right
    box_parts = ax.boxplot(
        [metric_data[metric_data['sae_display'] == sae]['value'].values for sae in sae_types],
        positions=np.arange(len(sae_types)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1},
        boxprops={'facecolor': 'lightgray', 'alpha': 0.7},
        whiskerprops={'linewidth': 0.8},
        capprops={'linewidth': 0.8}
    )
    
    # 3. Strip plot (individual points) on the right side
    for i, sae in enumerate(sae_types):
        sae_data = metric_data[metric_data['sae_display'] == sae]['value'].values
        n_points = len(sae_data)
        
        # Downsample if too many points
        if n_points > max_points:
            sampled_indices = np.random.choice(n_points, max_points, replace=False)
            sae_data_plot = sae_data[sampled_indices]
            # Add annotation about sampling
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(i, y_pos, f'({max_points}/{n_points})', 
                   ha='center', va='top', fontsize=8, style='italic', color='gray')
        else:
            sae_data_plot = sae_data
        
        # Add jittered points, truncated to avoid overlap with boxplot
        x_jitter = truncnorm.rvs(jitter_low, jitter_up, loc=i + RIGHT_OFFSET, scale=JITTER_STD, size=len(sae_data_plot))
        ax.scatter(x_jitter, sae_data_plot, 
                  alpha=0.4, s=3, color=colors[i], zorder=1, rasterized=True)
    
    # # Add sample size annotations
    # for i, sae in enumerate(sae_types):
    #     n = len(metric_data[metric_data['sae_display'] == sae])
    #     y_pos = ax.get_ylim()[0]
    #     ax.text(i, y_pos, f'n={n}', 
    #            ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Get display name for metric
    metric_display = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    # Formatting
    ax.set_xticks(np.arange(len(sae_types)))
    ax.set_xticklabels(sae_types, rotation=0, ha='center')
    ax.set_ylabel('')
    
    # Set title (with optional subtitle for filename)
    if filename:
        ax.set_title(f'{metric_display}\n{filename}', pad=4)
    else:
        ax.set_title(f'{metric_display}', pad=4)
    
    # Grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    if metric_name != 'num_children':
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    # Adjust x-axis limits to accommodate the layout
    ax.set_xlim(-0.6, len(sae_types) - 0.4)

    # ax.set_ylim(top=2100)  # Hardcode option for one edge case
    
    if created_fig:
        plt.tight_layout()
    
    return ax


def visualize_distributions(distributions_df, selected_metrics=None, output_path=None, 
                           figsize=(TEXTWIDTH_INCHES, TEXTWIDTH_INCHES / 1.618), plot_type='violin', filename=None, combine_plots=False):
    """
    Main function to create distribution visualizations.
    Can create either separate plots per metric or a single 4x2 subplot figure.
    
    Args:
        distributions_df: Distribution comparison DataFrame
        selected_metrics: Optional list of metrics to plot
        output_path: Optional path/directory to save figures (or single file if combine_plots=True)
        figsize: Figure size (width, height) - for individual plots or entire 4x2 grid if combine_plots=True
        plot_type: Type of plot to create ('violin' or 'raincloud')
        filename: Optional filename to display on plots
        combine_plots: If True, create single 4x2 subplot figure; if False, create separate plots
    """
    # Check if data is empty
    if distributions_df.empty:
        print("Error: Distribution dataframe is empty")
        return
    
    print(f"Loaded {len(distributions_df)} metric entries")
    print(f"SAE types: {sorted(distributions_df['sae_type'].unique())}")
    print(f"Metrics: {sorted(distributions_df['metric_name'].unique())}")
    
    # Prepare data for plotting
    print("\nTransforming to long format...")
    plot_data = prepare_long_format(distributions_df, selected_metrics)
    
    print(f"Expanded to {len(plot_data)} individual values")
    
    # Get unique metrics, ordered by METRICS list (same order as threshold sweep plots)
    available = set(plot_data['metric_name'].unique())
    metrics = [m for m in METRICS if m in available]
    metrics += sorted(available - set(metrics))
    
    if combine_plots:
        # Create single figure with 4x2 subplots
        print(f"\nCreating combined {plot_type} plot with {len(metrics)} subplots (4x2 layout)...")
        
        # Warn if more than 8 metrics
        if len(metrics) > 8:
            print(f"WARNING: {len(metrics)} metrics found, but only first 8 will be plotted in 4x2 layout")
            metrics = metrics[:8]
        
        # Create figure with 4x2 subplots
        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=COMBINED_FIGSIZE)
        axes = axes.flatten()
        # Add overall title if filename provided
        if filename:
            fig.suptitle(f'Distribution Comparisons: {filename}', fontsize=16, fontweight='bold', y=0.995)
        
        # Create one subplot per metric
        for idx, metric in enumerate(metrics):
            # Filter data for this metric
            metric_data = plot_data[plot_data['metric_name'] == metric]
            
            print(f"  Creating subplot {idx+1}/{len(metrics)} for: {metric}")
            
            # Create visualization based on plot type on the specific subplot
            if plot_type == 'raincloud':
                create_raincloud_plot_for_metric(
                    metric_data, metric,
                    filename=None, ax=axes[idx]
                )
            else:  # default to violin
                create_violin_plot_for_metric(
                    metric_data, metric,
                    filename=None, ax=axes[idx]
                )
        # Hide unused subplots if fewer than 8 metrics
        for idx in range(len(metrics), 8):
            axes[idx].set_visible(False)

        # Post-process: remove redundant axis labels/ticks across subplots
        n_cols = 2
        n_used = len(metrics)
        last_row = (n_used - 1) // n_cols
        for idx in range(n_used):
            ax = axes[idx]
            row, col = divmod(idx, n_cols)
            if row < last_row:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
                ax.set_xticklabels([])
            if col != 0:
                ax.set_ylabel('')

        # Single shared color legend at bottom
        available_sae_keys = [st for st in SAE_TYPES if st in set(plot_data['sae_type'].unique())]
        legend_handles = [
            mpatches.Patch(facecolor=SAE_COLORS.get(k, '#333333'), label=SAE_TYPE_DISPLAY_NAMES.get(k, k))
            for k in available_sae_keys
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
                   bbox_to_anchor=(0.5, 0), framealpha=0.9)
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        # Save figure if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
            print(f"\nSaved combined figure to {output_path}")
        
        plt.show()
        plt.close(fig)
    
    else:
        # Create separate plots (original behavior)
        print(f"\nCreating {len(metrics)} separate {plot_type} plots...")
        
        # Determine output directory if output_path is provided
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix:  # If it's a file path with extension
                output_dir = output_path.parent
                base_name = output_path.stem
                extension = output_path.suffix
            else:  # If it's a directory
                output_dir = output_path
                base_name = plot_type
                extension = ".pdf"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create one plot per metric
        for metric in metrics:
            # Filter data for this metric
            metric_data = plot_data[plot_data['metric_name'] == metric]
            
            print(f"\n  Creating {plot_type} plot for: {metric}")
            
            # Create visualization based on plot type
            if plot_type == 'raincloud':
                ax = create_raincloud_plot_for_metric(metric_data, metric, figsize=figsize, filename=filename)
            else:  # default to violin
                ax = create_violin_plot_for_metric(metric_data, metric, figsize=figsize, filename=filename)
            
            fig = ax.get_figure()
            
            # Save figure if output path provided
            if output_path:
                metric_filename = f"{base_name}_{metric}{extension}"
                metric_path = output_dir / metric_filename
                plt.savefig(metric_path, dpi=300, bbox_inches='tight')
                plt.savefig(metric_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
                print(f"    Saved to {metric_path}")
            
            plt.show()
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize distribution comparison dataframe as violin plots'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to distribution comparison pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output. If --combine-plots: single file path. Otherwise: directory or file base name'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='*',
        default=None,
        help='Optional: specific metrics to plot (if not provided, all metrics are plotted)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[TEXTWIDTH_INCHES, TEXTWIDTH_INCHES / 1.618],
        help='Figure size (width height) in inches. For --combine-plots mode, consider e.g. textwidth x 2*textwidth'
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['violin', 'raincloud'],
        default='raincloud',
        help='Type of plot to create (default: violin)'
    )
    parser.add_argument(
        '--max-points',
        type=int,
        default=7000,
        help='Maximum points to show per SAE type in raincloud plot (default: 7000)'
    )
    parser.add_argument(
        '--show-filename',
        action='store_true',
        help='Display the input filename as a subtitle on the plots'
    )
    parser.add_argument(
        '--combine-plots',
        action='store_true',
        help='Create a single figure with 4x2 subplots instead of separate plots per metric'
    )
    
    args = parser.parse_args()
    
    # Load distribution dataframe
    print(f"Loading distribution dataframe from {args.input}...")
    distributions_df = pd.read_pickle(args.input)

    if 'config_name' not in distributions_df.columns:
        raise KeyError(
            "Expected column 'config_name' in distribution dataframe for output-path metadata summary"
        )

    config_summary = summarize_config_name_components(distributions_df['config_name'])
    print(f"SAE types found: {config_summary['sae_types']}")
    print(f"Graph methods found: {config_summary['graph_methods']}")
    print(f"Foundation models found: {config_summary['foundation_models']}")

    if args.output:
        args.output = append_config_summary_to_output_path(args.output, config_summary)
        print(f"Output path updated to include config info: {args.output}")
    
    # Prepare filename for display if requested
    display_filename = None
    if args.show_filename:
        display_filename = Path(args.input).stem
    
    # Create visualization
    visualize_distributions(
        distributions_df,
        selected_metrics=args.metrics,
        output_path=args.output,
        figsize=tuple(args.figsize),
        plot_type=args.plot_type,
        filename=display_filename,
        combine_plots=args.combine_plots
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
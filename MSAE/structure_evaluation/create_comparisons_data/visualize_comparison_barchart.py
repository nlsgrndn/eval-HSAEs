"""
Script to visualize comparison dataframes as bar charts.

This script loads a comparison dataframe and creates bar chart visualizations
comparing metrics across different SAE types and graph creation methods.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from pathlib import Path
from structure_evaluation.create_comparisons_data.comparison_utils import (
    SAE_TYPES,
    SAE_COLORS,
    SAE_TYPE_DISPLAY_NAMES,
    GRAPH_METHODS,
    METRICS,
    METRIC_DISPLAY_NAMES,
    TEXTWIDTH_INCHES,
    extract_foundation_model,
    extract_sae_type,
    extract_graph_method,
    extract_mean_centering_mode,
    summarize_config_name_components,
    append_config_summary_to_output_path,
)

# Control whether to show standard deviation error bars
# Set to False globally to disable all error bars, or specify per-metric
SHOW_STD = True

# Metrics for which standard deviation should NOT be shown
METRICS_WITHOUT_STD = set([
    # Add metric names here if you want to disable std for specific metrics
    # Example: 'coverage',
    "child_greater_than_parent",
    "ms_child_greater_than_parent",
])


GRAPH_METHOD_DISPLAY_NAMES = {
    'CONDACT': 'CondAct',
    'MCS': 'MCS',
    'WONDACT': 'WONDACT',
}

GRAPH_METHOD_HATCHES = {
    'CONDACT': None,
    'MCS': '//',
    'WONDACT': '\\\\',
}

GRAPH_METHOD_TITLES = {
    'CONDACT': 'Conditional Activation Graph Creation',
    'MCS': 'Masked Cosine Similarity Graph Creation',
    'WONDACT': 'WONDACT Graph Creation',
}


def _get_present_graph_methods(method_values):
    """Return supported graph methods in canonical order."""
    present = set(method_values)
    return [method for method in GRAPH_METHODS if method in present]


def prepare_data_for_plotting(comparison_df, metrics, graph_method=None, mc_mode=None):
    """
    Prepare data for bar chart plotting.
    
    Args:
        comparison_df: DataFrame with comparison data
        metrics: List of metric names to plot
        graph_method: Graph creation method to filter by, or None to skip filter
        mc_mode: Mean centering mode to filter by ('MC' or 'non-MC'), or None to skip filter
    
    Returns:
        DataFrame with columns: sae_type, mc_mode, metric, mean, std
    """
    comparison_df = comparison_df.copy()
    comparison_df['sae_type'] = comparison_df['config_name'].apply(extract_sae_type)
    comparison_df['graph_method'] = comparison_df['config_name'].apply(extract_graph_method)
    comparison_df['mc_mode'] = comparison_df['config_name'].apply(extract_mean_centering_mode)
    
    filtered_df = comparison_df
    if graph_method is not None:
        filtered_df = filtered_df[filtered_df['graph_method'] == graph_method]
        if len(filtered_df) == 0:
            print(f"Warning: No data found for graph method '{graph_method}'")
            return pd.DataFrame()
    if mc_mode is not None:
        filtered_df = filtered_df[filtered_df['mc_mode'] == mc_mode]
        if len(filtered_df) == 0:
            print(f"Warning: No data found for MC mode '{mc_mode}'")
            return pd.DataFrame()
    
    # Prepare data for plotting
    plot_data = []
    for _, row in filtered_df.iterrows():
        sae_type = row['sae_type']
        row_mc_mode = row['mc_mode']
        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col not in row.index:
                print(f"Warning: Metric '{metric}' not found in dataframe")
                continue
            
            mean_val = row[mean_col]
            std_val = row[std_col] if std_col in row.index else 0
            
            plot_data.append({
                'sae_type': sae_type,
                'mc_mode': row_mc_mode,
                'graph_method': row['graph_method'],
                'metric': metric,
                'mean': mean_val,
                'std': std_val
            })
    
    return pd.DataFrame(plot_data)


def create_bar_chart(ax, plot_data, metrics, title, ylabel='Value', sae_types=None,
                     show_xticklabels=True, group_col='sae_type',
                     group_labels=None, group_colors=None, group_order=None,
                     group_alphas=None, group_hatches=None,
                     split_legend_sae_handles=None, split_legend_variation_handles=None):
    """
    Create a grouped bar chart.
    
    Args:
        ax: Matplotlib axis
        plot_data: DataFrame with columns [group_col, metric, mean, std]
        metrics: List of metrics to plot
        title: Chart title
        ylabel: Y-axis label
        sae_types: Deprecated, use group_order instead
        show_xticklabels: Whether to show x-axis tick labels
        group_col: Column in plot_data to group bars by (default 'sae_type')
        group_labels: Dict mapping group key -> display name (fallback: SAE_TYPE_DISPLAY_NAMES)
        group_colors: Dict mapping group key -> color (fallback: SAE_COLORS)
        group_order: Ordered list of group keys to use (fallback: SAE_TYPES order for sae_type col)
        group_alphas: Dict mapping group key -> alpha value (fallback: 0.8)
        group_hatches: Dict mapping group key -> hatch pattern string or None
        split_legend_sae_handles: List of (handle, label) for SAE types; if provided together with
            split_legend_variation_handles a compact split legend is drawn instead of
            the default per-bar legend.
        split_legend_variation_handles: List of (handle, label) for the variation dimension
            (MC mode or graph method). Used together with split_legend_sae_handles.
    """
    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    if group_order is None:
        if group_col == 'sae_type':
            available = set(plot_data[group_col].unique())
            group_order = [st for st in SAE_TYPES if st in available]
        else:
            group_order = list(plot_data[group_col].unique())
    
    if group_labels is None:
        group_labels = SAE_TYPE_DISPLAY_NAMES
    if group_colors is None:
        group_colors = SAE_COLORS
    
    n_metrics = len(metrics)
    n_groups = len(group_order)
    
    bar_width = 0.8 / n_groups
    x = np.arange(n_metrics)
    
    for i, group_key in enumerate(group_order):
        group_data = plot_data[plot_data[group_col] == group_key]
        
        means = []
        stds = []
        for metric in metrics:
            metric_data = group_data[group_data['metric'] == metric]
            if len(metric_data) > 0:
                means.append(metric_data['mean'].values[0])
                stds.append(metric_data['std'].values[0])
            else:
                means.append(np.nan)
                stds.append(0)
        
        positions = x + (i - n_groups / 2 + 0.5) * bar_width
        
        stds_to_show = [
            std if (SHOW_STD and metric not in METRICS_WITHOUT_STD) else np.nan
            for metric, std in zip(metrics, stds)
        ]
        
        display_name = group_labels.get(group_key, group_key)
        color = group_colors.get(group_key, '#333333')
        alpha = group_alphas.get(group_key, 0.8) if group_alphas else 0.8
        hatch = group_hatches.get(group_key, None) if group_hatches else None
        # Hatched bars: dark grey edge so the pattern is visible.
        # Solid bars: edge matches face for a clean look.
        edgecolor = '#555555' if hatch else color
        # Error bar cap width proportional to bar width (in data units)
        capsize = bar_width * 30
        bars = ax.bar(positions, means, bar_width, label=display_name,
                      color=color, alpha=alpha,
                      hatch=hatch, edgecolor=edgecolor, linewidth=0.5)
        if SHOW_STD:
            ax.errorbar(positions, means, yerr=stds_to_show,
                        fmt='none', ecolor='#333333', elinewidth=0.8,
                        capsize=capsize * 0.5, capthick=0.8) # set 0.5 to set cap width relative to bar width
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    display_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    
    if show_xticklabels:
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    else:
        ax.set_xticklabels([])

    # Build legend: either split (SAE types + variation) or default per-bar
    if split_legend_sae_handles is not None and split_legend_variation_handles is not None:
        import matplotlib.patches as mpatches
        separator = mpatches.Patch(visible=False, label='')  # invisible spacer
        all_handles = (
            [h for h, _ in split_legend_sae_handles]
            + [separator]
            + [h for h, _ in split_legend_variation_handles]
        )
        all_labels = (
            [l for _, l in split_legend_sae_handles]
            + ['']
            + [l for _, l in split_legend_variation_handles]
        )
        n_cols = len(split_legend_sae_handles) + 1 + len(split_legend_variation_handles)
        ax.legend(handles=all_handles, labels=all_labels,
                  loc='upper center', ncol=n_cols, fontsize=10,
                  frameon=True, handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    else:
        n_groups = len(group_order) if group_order else 1
        ax.legend(loc='upper center', ncol=n_groups, fontsize=10,
                  frameon=True, handlelength=1.5, handletextpad=0.5, columnspacing=1.0)

    ax.set_ylim(top=1.19)
    yticks = np.arange(0.0, 1.19 + 1e-9, 0.2)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)


def _build_mc_group_keys(comparison_df, variant_outer=True):
    """
    Build (sae_type, mc_mode) combo group keys for MC-mode comparison.

    MC bars are solid (full alpha); non-MC bars are lighter.

    Args:
        variant_outer: If True (default), show all SAE types for each MC mode together
            (non-interleaved: MC block then non-MC block). If False, interleave MC/non-MC
            within each SAE type.

    Returns:
        group_order, group_labels, group_colors, group_alphas, group_hatches,
        sae_legend_handles, variation_legend_handles
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    available_sae_types = set(comparison_df['sae_type'].unique())
    sae_types_ordered = [st for st in SAE_TYPES if st in available_sae_types]
    mc_modes = ['MC', 'non-MC']

    ALPHA_LOW = 0.55

    group_order = []
    group_labels = {}
    group_colors = {}
    group_alphas = {}
    group_hatches = {}

    outer = mc_modes if variant_outer else sae_types_ordered
    inner = sae_types_ordered if variant_outer else mc_modes

    for outer_val in outer:
        for inner_val in inner:
            sae_type = inner_val if variant_outer else outer_val
            mc_mode  = outer_val if variant_outer else inner_val
            base_color = mcolors.to_rgb(SAE_COLORS.get(sae_type, '#333333'))
            key = f"{sae_type}__{mc_mode}"
            if key not in group_labels:  # avoid duplicates
                group_order.append(key)
            group_labels[key] = f"{SAE_TYPE_DISPLAY_NAMES.get(sae_type, sae_type)} ({mc_mode})"
            group_colors[key] = base_color
            # MC mode: distinguish only via alpha (solid, full vs lighter)
            group_alphas[key] = 0.85 if mc_mode == 'MC' else ALPHA_LOW
            group_hatches[key] = None

    # SAE type legend handles: colored patch per SAE type
    sae_legend_handles = [
        (mpatches.Patch(facecolor=mcolors.to_rgb(SAE_COLORS.get(st, '#333333')),
                        alpha=0.85, edgecolor='none'),
         SAE_TYPE_DISPLAY_NAMES.get(st, st))
        for st in sae_types_ordered
    ]
    # Variation legend handles: show alpha difference using neutral grey
    _grey = (0.5, 0.5, 0.5)
    variation_legend_handles = [
        (mpatches.Patch(facecolor=_grey, alpha=0.85, edgecolor='none'), 'MC'),
        (mpatches.Patch(facecolor=_grey, alpha=ALPHA_LOW,  edgecolor='none'), 'non-MC'),
    ]

    return (group_order, group_labels, group_colors, group_alphas, group_hatches,
            sae_legend_handles, variation_legend_handles)


def _build_graph_method_group_keys(comparison_df, variant_outer=True):
    """
    Build (sae_type, graph_method) combo group keys for single-plot graph-method comparison.

    CONDACT bars are solid; MCS and WONDACT use distinct hatching.

    Args:
        variant_outer: If True (default), show all SAE types for each graph method together
            (non-interleaved by method blocks). If False, interleave within each
            SAE type.

    Returns:
        group_order, group_labels, group_colors, group_alphas, group_hatches,
        sae_legend_handles, variation_legend_handles
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    available_sae_types = set(comparison_df['sae_type'].unique())
    sae_types_ordered = [st for st in SAE_TYPES if st in available_sae_types]
    available_methods = _get_present_graph_methods(comparison_df['graph_method'].unique())

    group_order = []
    group_labels = {}
    group_colors = {}
    group_alphas = {}
    group_hatches = {}

    outer = available_methods if variant_outer else sae_types_ordered
    inner = sae_types_ordered if variant_outer else available_methods

    for outer_val in outer:
        for inner_val in inner:
            sae_type = inner_val if variant_outer else outer_val
            method   = outer_val if variant_outer else inner_val
            base_color = mcolors.to_rgb(SAE_COLORS.get(sae_type, '#333333'))
            key = f"{sae_type}__{method}"
            if key not in group_labels:  # avoid duplicates
                group_order.append(key)
            method_display = GRAPH_METHOD_DISPLAY_NAMES.get(method, method)
            group_labels[key] = f"{SAE_TYPE_DISPLAY_NAMES.get(sae_type, sae_type)} ({method_display})"
            group_colors[key] = base_color
            # Graph method: distinguish via hatch while keeping SAE colors consistent.
            group_alphas[key] = 0.85
            group_hatches[key] = GRAPH_METHOD_HATCHES.get(method, '//')

    # SAE type legend handles
    sae_legend_handles = [
        (mpatches.Patch(facecolor=mcolors.to_rgb(SAE_COLORS.get(st, '#333333')),
                        alpha=0.85, edgecolor='none'),
         SAE_TYPE_DISPLAY_NAMES.get(st, st))
        for st in sae_types_ordered
    ]
    # Variation legend handles: show method hatch differences using neutral grey.
    _grey = (0.5, 0.5, 0.5)
    variation_legend_handles = []
    for method in available_methods:
        hatch = GRAPH_METHOD_HATCHES.get(method, '//')
        edgecolor = '#555555' if hatch else 'none'
        variation_legend_handles.append(
            (
                mpatches.Patch(facecolor=_grey, alpha=0.85, hatch=hatch, edgecolor=edgecolor),
                GRAPH_METHOD_DISPLAY_NAMES.get(method, method),
            )
        )

    return (group_order, group_labels, group_colors, group_alphas, group_hatches,
            sae_legend_handles, variation_legend_handles)


def visualize_comparison(comparison_df, output_path=None, figsize=(TEXTWIDTH_INCHES, TEXTWIDTH_INCHES * 1.2),
                         split_subplots=False, variant_outer=True):
    """
    Create visualization comparing SAE types across either graph creation methods or
    mean-centering modes (whichever varies in the data — never both simultaneously).

    - When MC mode varies: bars grouped by (SAE type × MC mode); MC=solid, non-MC=lighter.
      One subplot per graph method present.
    - When graph method varies and split_subplots=False (default): single plot with bars grouped
            by (SAE type × graph method); methods are distinguished by hatch style.
    - When graph method varies and split_subplots=True: one subplot per method, bars by SAE type.

    Args:
        comparison_df: DataFrame with comparison data
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        split_subplots: If True and graph method varies, use one subplot per method instead of
                        a single combined plot (default False)
        variant_outer: If True (default), show all SAE types for each variant grouped together
                       (non-interleaved). If False, interleave variants within each SAE type.
    """
    comparison_df = comparison_df.copy()
    comparison_df['graph_method'] = comparison_df['config_name'].apply(extract_graph_method)
    comparison_df['mc_mode'] = comparison_df['config_name'].apply(extract_mean_centering_mode)
    comparison_df['sae_type'] = comparison_df['config_name'].apply(extract_sae_type)

    available_methods = _get_present_graph_methods(comparison_df['graph_method'].unique())
    available_mc_modes = comparison_df['mc_mode'].unique().tolist()

    mc_varies = 'MC' in available_mc_modes and 'non-MC' in available_mc_modes

    # -------------------------------------------------------------------------
    # Case 1: MC mode varies — bars grouped by (sae_type, mc_mode)
    # -------------------------------------------------------------------------
    if mc_varies:
        graph_methods_present = available_methods
        n_subplots = len(graph_methods_present)
        if n_subplots == 0:
            print("Warning: No supported graph methods found in dataframe")
            return

        if n_subplots > 1:
            fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=False, sharey=True)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] / 2))
            axes = [axes]

        group_order, group_labels, group_colors, group_alphas, group_hatches, \
            sae_legend_handles, variation_legend_handles = \
            _build_mc_group_keys(comparison_df, variant_outer=variant_outer)

        for idx, method in enumerate(graph_methods_present):
            ax = axes[idx]
            plot_data = prepare_data_for_plotting(comparison_df, METRICS, graph_method=method)
            # Add group key column combining sae_type and mc_mode
            if len(plot_data) > 0:
                plot_data['group_key'] = plot_data['sae_type'] + '__' + plot_data['mc_mode']
            show_labels = (idx == n_subplots - 1)
            create_bar_chart(ax, plot_data, METRICS,
                             None,
                             ylabel='Value',
                             show_xticklabels=show_labels,
                             group_col='group_key',
                             group_labels=group_labels,
                             group_colors=group_colors,
                             group_order=group_order,
                             group_alphas=group_alphas,
                             group_hatches=group_hatches,
                             split_legend_sae_handles=sae_legend_handles,
                             split_legend_variation_handles=variation_legend_handles)

    # -------------------------------------------------------------------------
    # Case 2: Graph method varies
    # -------------------------------------------------------------------------
    else:
        if not available_methods:
            print("Warning: No supported graph methods found in dataframe")
            return

        # Single-plot path: all available methods on one plot, grouped by (sae_type, graph_method)
        if len(available_methods) > 1 and not split_subplots:
            fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] / 2))
            group_order, group_labels, group_colors, group_alphas, group_hatches, \
                sae_legend_handles, variation_legend_handles = \
                _build_graph_method_group_keys(comparison_df, variant_outer=variant_outer)
            plot_data = prepare_data_for_plotting(comparison_df, METRICS)
            if len(plot_data) > 0:
                plot_data['group_key'] = plot_data['sae_type'] + '__' + plot_data['graph_method']
            create_bar_chart(ax, plot_data, METRICS,
                             None,
                             ylabel='Value',
                             show_xticklabels=True,
                             group_col='group_key',
                             group_labels=group_labels,
                             group_colors=group_colors,
                             group_order=group_order,
                             group_alphas=group_alphas,
                             group_hatches=group_hatches,
                             split_legend_sae_handles=sae_legend_handles,
                             split_legend_variation_handles=variation_legend_handles)

        # Split-subplots path (or only one method present): one subplot per method.
        else:
            n_subplots = len(available_methods)
            if n_subplots > 1:
                fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=False, sharey=True)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] / 2))
                axes = [ax]

            if n_subplots > 1:
                axes = np.atleast_1d(axes)

            for idx, method in enumerate(available_methods):
                method_data = prepare_data_for_plotting(comparison_df, METRICS, graph_method=method)
                # foundation_models = comparison_df['config_name'].apply(extract_foundation_model).unique().tolist()
                # # export to csv
                # assert len(foundation_models) == 1, "Expected exactly one foundation model in graph-method comparison"
                # import os
                # os.makedirs('experiments_results', exist_ok=True)
                # method_data.to_csv(f'experiments_results/barchart_plot_data_{foundation_models[0]}.csv', index=False)
                show_labels = (idx == n_subplots - 1)
                create_bar_chart(axes[idx], method_data, METRICS,
                                 None,
                                 ylabel='Value',
                                 show_xticklabels=show_labels)

    if output_path:
        # Reserve a small footer area and print output path in very small text.
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        fig.text(0.5, 0.005, str(output_path), ha='center', va='bottom',
                 fontsize=5, color='#666666')
    else:
        fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize comparison dataframe as bar charts')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to comparison dataframe pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output figure (e.g., comparison_barchart.png)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[TEXTWIDTH_INCHES, TEXTWIDTH_INCHES * 1.2],
        help='Figure size (width height) in inches'
    )
    parser.add_argument(
        '--split-subplots',
        action='store_true',
        default=False,
        help='When graph method varies, use one subplot per method instead of a single combined plot'
    )
    parser.add_argument(
        '--interleaved',
        action='store_true',
        default=False,
        help='Interleave variants within each SAE type (default: grouped by variant)'
    )

    args = parser.parse_args()
    
    # Load comparison dataframe
    print(f"Loading comparison dataframe from {args.input}...")
    comparison_df = pd.read_pickle(args.input)
    print(f"Loaded {len(comparison_df)} configurations")
    
    config_summary = summarize_config_name_components(comparison_df['config_name'])

    print(f"\nSAE types found: {config_summary['sae_types']}")
    print(f"Graph methods found: {config_summary['graph_methods']}")
    print(f"Foundation models found: {config_summary['foundation_models']}")

    # add the sae_types, graph_methods, and foundation_models to the output path for better organization
    if args.output:
        args.output = append_config_summary_to_output_path(args.output, config_summary)
        print(f"Output path updated to include config info: {args.output}")

    # Create visualization
    print("\nCreating visualization...")
    visualize_comparison(comparison_df, output_path=args.output, figsize=tuple(args.figsize),
                         split_subplots=args.split_subplots, variant_outer=not args.interleaved)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

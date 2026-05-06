"""
Script to compare num_children distributions between MC and non-MC conditions.

Loads two distribution pickles (MC and nonMC), filters to num_children,
and produces a raincloud plot with interleaved MC/nonMC pairs per SAE type.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from pathlib import Path
from matplotlib.patches import Patch
from structure_evaluation.create_comparisons_data.comparison_utils import (
    SAE_TYPE_DISPLAY_NAMES,
    SAE_TYPES,
    SAE_COLORS,
    TEXTWIDTH_INCHES,
)

FIGSIZE = (TEXTWIDTH_INCHES, TEXTWIDTH_INCHES / 3)

BASE_FONT_SIZE = 7
SAE_TYPE_LABEL_FONT_SIZE = BASE_FONT_SIZE  - 1
SHOW_XTICK_LABELS = True
plt.rcParams.update({
    'font.size':        BASE_FONT_SIZE,
    'axes.titlesize':   BASE_FONT_SIZE - 1,
    'axes.labelsize':   BASE_FONT_SIZE + 1,
    'xtick.labelsize':  BASE_FONT_SIZE - 1,
    'ytick.labelsize':  BASE_FONT_SIZE - 1,
    'legend.fontsize':  BASE_FONT_SIZE - 1,
    'lines.linewidth':  1.0,
    'lines.markersize': 3,
})


def _lighten_hex(hex_color, factor=0.45):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


def _expand_values(df, condition):
    rows = []
    for _, row in df.iterrows():
        for val in row['values']:
            rows.append({'sae_type': row['sae_type'], 'condition': condition, 'value': val})
    return pd.DataFrame(rows)


def plot_num_children_mc_vs_nonmc(mc_df, nonmc_df, output_path=None, max_points=5000):
    """
    Raincloud plot of num_children comparing MC vs non-MC on a single axis.
    For each SAE type, MC and non-MC are shown as an adjacent pair.
    """
    mc_vals = mc_df[mc_df['metric_name'] == 'num_children']
    nonmc_vals = nonmc_df[nonmc_df['metric_name'] == 'num_children']

    plot_data = pd.concat(
        [_expand_values(mc_vals, 'MC'), _expand_values(nonmc_vals, 'non-MC')],
        ignore_index=True,
    )

    available_keys = set(plot_data['sae_type'].unique())
    ordered_keys = [st for st in SAE_TYPES if st in available_keys]
    ordered_keys += sorted(available_keys - set(ordered_keys))

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    LEFT_OFFSET = -0.15
    RIGHT_OFFSET = 0.2
    JITTER_STD = 0.04
    SAE_GROUP_GAP = 0.35
    jitter_low = -0.1 / JITTER_STD
    jitter_up = 0.1 / JITTER_STD

    entries = []
    for sae_key in ordered_keys:
        base_color = SAE_COLORS.get(sae_key, '#333333')
        entries.append((sae_key, 'MC', base_color))
        entries.append((sae_key, 'non-MC', _lighten_hex(base_color)))

    x_labels = [cond for _, cond, _ in entries]
    colors = [color for _, _, color in entries]
    data_per_entry = [
        plot_data.loc[
            (plot_data['sae_type'] == sae_key) & (plot_data['condition'] == cond), 'value'
        ].values
        for sae_key, cond, _ in entries
    ]
    positions = []
    pair_centers = []
    for sae_idx, _ in enumerate(ordered_keys):
        left_pos = sae_idx * (2 + SAE_GROUP_GAP)
        positions.extend([left_pos, left_pos + 1])
        pair_centers.append(left_pos + 0.5)
    positions = np.array(positions, dtype=float)

    # 1. Half-violin
    violin_parts = ax.violinplot(
        data_per_entry,
        positions=positions + LEFT_OFFSET,
        widths=0.5,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        vertices = pc.get_paths()[0].vertices
        vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, positions[i] + LEFT_OFFSET)

    # 2. Box plot
    ax.boxplot(
        data_per_entry,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1},
        boxprops={'facecolor': 'lightgray', 'alpha': 0.7},
        whiskerprops={'linewidth': 0.8},
        capprops={'linewidth': 0.8},
    )

    # 3. Strip plot
    for i, (vals, color) in enumerate(zip(data_per_entry, colors)):
        if len(vals) > max_points:
            vals = vals[np.random.choice(len(vals), max_points, replace=False)]
        x_jitter = truncnorm.rvs(
            jitter_low, jitter_up,
            loc=positions[i] + RIGHT_OFFSET,
            scale=JITTER_STD,
            size=len(vals),
        )
        ax.scatter(x_jitter, vals, alpha=0.55, s=3, color=color, zorder=1, rasterized=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.tick_params(axis='x', labelbottom=SHOW_XTICK_LABELS)
    if SHOW_XTICK_LABELS:
        for idx, sae_key in enumerate(ordered_keys):
            pair_center = pair_centers[idx]
            ax.text(
                pair_center,
                -0.22,
                SAE_TYPE_DISPLAY_NAMES.get(sae_key, sae_key),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=SAE_TYPE_LABEL_FONT_SIZE,
                clip_on=False,
            )
    ax.set_ylabel('NumberOfChildren')
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)

    legend_sae_keys = [st for st in SAE_TYPES if st in available_keys][:3]
    sae_handles = [
        Patch(
            facecolor=SAE_COLORS.get(sae_key, '#333333'),
            edgecolor='none',
            label=SAE_TYPE_DISPLAY_NAMES.get(sae_key, sae_key),
        )
        for sae_key in legend_sae_keys
    ]
    neutral_grey = '#808080'
    variation_handles = [
        Patch(facecolor=neutral_grey, edgecolor='none', alpha=0.85, label='MC'),
        Patch(facecolor=neutral_grey, edgecolor='none', alpha=0.55, label='non-MC'),
    ]
    separator = Patch(visible=False, label='')
    all_handles = sae_handles + [separator] + variation_handles
    all_labels = [h.get_label() for h in sae_handles] + [''] + [h.get_label() for h in variation_handles]
    ax.legend(
        handles=all_handles,
        labels=all_labels,
        loc='upper right',
        ncol=len(all_handles),
        frameon=True,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    plt.tight_layout(rect=[0, 0.16, 1, 1])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path.with_suffix('.pdf'), dpi=600, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}.png")
        print(f"Saved to {output_path}.pdf")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare num_children distributions: MC vs non-MC'
    )
    parser.add_argument('--mc-input', type=str, required=True,
                        help='Path to MC distributions pickle')
    parser.add_argument('--nonmc-input', type=str, required=True,
                        help='Path to nonMC distributions pickle')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (without extension)')
    parser.add_argument('--max-points', type=int, default=5000)

    args = parser.parse_args()

    print(f"Loading MC data from {args.mc_input}...")
    mc_df = pd.read_pickle(args.mc_input)

    print(f"Loading nonMC data from {args.nonmc_input}...")
    nonmc_df = pd.read_pickle(args.nonmc_input)

    plot_num_children_mc_vs_nonmc(
        mc_df, nonmc_df,
        output_path=args.output,
        max_points=args.max_points,
    )
    print("Done!")


if __name__ == "__main__":
    main()

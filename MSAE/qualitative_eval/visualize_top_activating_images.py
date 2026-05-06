"""
Visualize the top-4 activating images for a single SAE feature in a 2x2 grid.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from my_config import CONFIGS
from utils_sae_feature_properties import SAEDimensions
from structure_extraction.structure_extraction_utils import get_graph
from my_utils import load_dataset


def _center_crop_square(img):
    try:
        from PIL import Image as PILImage
        if isinstance(img, PILImage.Image):
            img = np.array(img)
    except ImportError:
        pass
    if hasattr(img, 'numpy'):
        img = img.permute(1, 2, 0).numpy() if img.ndim == 3 else img.numpy()
    img = np.array(img)
    h, w = img.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return img[top:top + s, left:left + s]


def visualize_top_activating_images(config, node_id, label=None, n=4, save_path=None):
    print("Loading top-k activating images...")
    topk_indices, _ = SAEDimensions(config).get_topk_activating_images_indices_and_values()

    indices = topk_indices[node_id][:n]

    print("Loading dataset...")
    dataset = load_dataset(config.graph_eval_dataset, False)

    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    title = label if label is not None else f"node {node_id}"
    # fig.suptitle(title, fontsize=30)
    # fig.subplots_adjust(top=0.88)

    for i, ax in enumerate(axes.flat):
        if i >= len(indices):
            ax.axis('off')
            continue
        img = dataset[indices[i]]
        if isinstance(img, tuple):
            img = img[0]
        ax.imshow(_center_crop_square(img))
        ax.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def main(config_start_str, node_id):
    config = None
    for config_name, cfg in CONFIGS.items():
        if config_name.startswith(config_start_str):
            config = cfg
            print(f"Selected config: {config_name}")
            break

    if config is None:
        raise ValueError(f"No config found starting with '{config_start_str}'")

    print("Loading graph for labels...")
    _, _subset_map, sae_ids, labels = get_graph(config)
    original_sae_id_to_label = dict(zip(sae_ids, labels))
    label = original_sae_id_to_label.get(node_id, f"node {node_id}")

    out_dir = "experiments_results/qualitative_evaluation/top_activating_images"
    save_path = os.path.join(out_dir, f"{config_start_str}_node{node_id}.pdf")

    visualize_top_activating_images(config, node_id, label=label, n=4, save_path=save_path)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize top-4 activating images for a SAE feature")
    parser.add_argument("--config_start_str", type=str, default="msae_rw",
                        help="Prefix of config name to select from CONFIGS")
    parser.add_argument("--node_id", "-n", type=int, required=True,
                        help="SAE feature / node ID to visualize")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_start_str=args.config_start_str, node_id=args.node_id)

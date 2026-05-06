"""
Full subDAG structure visualization with tree-aware layout.
Root at top; leaves at bottom. Each node shows images from its
associated_original_sae_ids. Arrows connect parent to child.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx

from my_config import CONFIGS
from utils_sae_feature_properties import SAEDimensions
from structure_extraction.structure_extraction_utils import get_graph
from my_utils import load_dataset


def _center_crop_square(img):
    """Center-crop img (PIL, tensor, or ndarray) to a square without changing aspect ratio."""
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


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _compute_dag_levels(G, root_node_id):
    """Longest-path depth from root so every parent sits strictly above its children."""
    levels = {root_node_id: 0}
    for node in nx.topological_sort(G):
        if node not in levels:
            continue
        for child in G.successors(node):
            d = levels[node] + 1
            if child not in levels or levels[child] < d:
                levels[child] = d
    return levels


def _assign_x_slots(subG, root_node_id, levels, level_nodes):
    """
    Tree-aware x-positions (unit = 1 slot):

    1. DFS post-order from root: leaf nodes get sequential slots 0, 1, 2, ...
       so sibling subtrees never overlap.
    2. Internal nodes sit at the arithmetic mean of their children's slots.
    3. Left-sweep per level resolves any remaining overlaps (minimum
       separation = 1 slot), preserving the relative left-to-right order.

    DAG diamonds are handled naturally: a shared child keeps the slot
    assigned on first DFS visit; its multiple parents each centre over
    their own child sets independently.
    """
    n_levels = max(levels.values()) + 1
    x: dict[int, float] = {}
    _visited: set = set()
    _ctr = [0]

    def _dfs(node):
        if node in _visited:
            return
        _visited.add(node)
        for child in subG.successors(node):
            _dfs(child)
        if not list(subG.successors(node)):          # leaf
            x[node] = float(_ctr[0])
            _ctr[0] += 1

    _dfs(root_node_id)
    n_leaves = max(_ctr[0], 1)

    # Internal nodes: centroid of children (process bottom-up)
    for lv in range(n_levels - 1, -1, -1):
        for node in level_nodes.get(lv, []):
            if node not in x:
                children_x = [x[c] for c in subG.successors(node) if c in x]
                x[node] = np.mean(children_x) if children_x else (n_leaves - 1) / 2.0

    # Resolve overlaps: left-sweep per level
    for lv in range(n_levels):
        row = sorted(level_nodes.get(lv, []), key=lambda n: x.get(n, 0.0))
        for i in range(1, len(row)):
            if x[row[i]] < x[row[i - 1]] + 1.0:
                x[row[i]] = x[row[i - 1]] + 1.0

    return x


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def create_dag_visualization(G, root_node_id, topk_indices, dataset, original_sae_id_to_label,
                              image_size=2.0, max_images_per_node=3, max_depth=None,
                              h_gap=0.5, v_gap=1.4, save_path=None):
    """
    Visualize the subDAG rooted at root_node_id with a tree-aware layout.

    Layout:
        Leaf nodes are placed left-to-right in DFS traversal order.
        Internal nodes are centred over their children.
        Per-level overlap resolution guarantees no two nodes collide.

    Each node renders as a horizontal strip of images, one per
    associated_original_sae_id (up to max_images_per_node), with a light
    background panel behind the strip.

    Args:
        G: Directed graph, edges go parent → child.
        root_node_id: Root of the subDAG to visualise.
        topk_indices: {original_sae_id: [image_idx, ...]} top-activating indices.
        dataset: Indexable image dataset.
        original_sae_id_to_label: {original_sae_id: label_string}.
        image_size: Width/height of one image in inches.
        max_images_per_node: Max SAE features shown per graph node.
        max_depth: Clip the DAG at this depth from the root (None = full).
        h_gap: Horizontal clearance between adjacent nodes (inches).
        v_gap: Vertical clearance between levels (inches).
        save_path: Optional path to save the figure.
    """
    # --- Build subgraph --------------------------------------------------
    all_nodes = {root_node_id} | nx.descendants(G, root_node_id)
    subG = G.subgraph(all_nodes).copy()

    levels = _compute_dag_levels(subG, root_node_id)

    if max_depth is not None:
        all_nodes = {n for n, lv in levels.items() if lv <= max_depth}
        subG = subG.subgraph(all_nodes).copy()
        levels = {n: lv for n, lv in levels.items() if n in all_nodes}

    n_levels = max(levels.values()) + 1
    level_nodes: dict = {}
    for node, lv in levels.items():
        level_nodes.setdefault(lv, []).append(node)

    print(f"SubDAG: {len(all_nodes)} nodes | {subG.number_of_edges()} edges | {n_levels} levels")

    # --- Node geometry ---------------------------------------------------
    img_gap = 0.12   # gap between images within one node strip

    def _strip_w(node_id):
        n = max(min(len(G.nodes[node_id].get('associated_original_sae_ids', [])),
                    max_images_per_node), 1)
        return n * image_size + (n - 1) * img_gap

    max_strip_w = max(_strip_w(n) for n in all_nodes)
    node_h = image_size + 0.5          # image + title space
    slot_w = max_strip_w + h_gap       # one layout column = widest node + gap

    # --- Compute x positions (tree layout) -------------------------------
    x_slots = _assign_x_slots(subG, root_node_id, levels, level_nodes)

    # Convert slots → inches; centre the whole layout horizontally
    x_inch = {n: s * slot_w for n, s in x_slots.items()}
    x_min = min(x_inch.values()) - max_strip_w / 2
    x_max = max(x_inch.values()) + max_strip_w / 2
    margin = h_gap / 2
    total_width = x_max - x_min + 2 * margin
    x_inch = {n: x - x_min + margin for n, x in x_inch.items()}

    # --- Compute y positions (root at top) -------------------------------
    total_height = n_levels * (node_h + v_gap) + v_gap

    def _cy(lv):
        return total_height - v_gap / 2 - lv * (node_h + v_gap) - node_h / 2

    node_pos = {n: (x_inch[n], _cy(levels[n])) for n in all_nodes}

    # --- Figure ----------------------------------------------------------
    fig = plt.figure(figsize=(total_width, total_height), facecolor='white')

    # Draw edges (behind everything else)
    for src, dst in subG.edges():
        if src not in node_pos or dst not in node_pos:
            continue
        sx, sy = node_pos[src]
        dx, dy = node_pos[dst]
        arrow = FancyArrowPatch(
            (sx / total_width,  (sy - node_h / 2) / total_height),
            (dx / total_width,  (dy + node_h / 2) / total_height),
            transform=fig.transFigure,
            arrowstyle='->,head_width=0.25,head_length=0.25',
            color='#aaaaaa', linewidth=1.0, alpha=0.7, zorder=1,
        )
        fig.add_artist(arrow)

    # Draw nodes
    for nid, (cx, cy) in node_pos.items():
        sae_ids = G.nodes[nid].get('associated_original_sae_ids', [])[:max_images_per_node]
        if not sae_ids:
            continue

        n_imgs = len(sae_ids)
        strip_w = n_imgs * image_size + (n_imgs - 1) * img_gap
        ix0 = cx - strip_w / 2      # left edge of image strip
        iy0 = cy - image_size / 2   # bottom edge of images

        for i, sae_id in enumerate(sae_ids):
            img = dataset[topk_indices[sae_id][0]]
            if isinstance(img, tuple):
                img = img[0]
            ix = ix0 + i * (image_size + img_gap)
            ax = fig.add_axes([
                ix / total_width,
                iy0 / total_height,
                image_size / total_width,
                image_size / total_height,
            ])
            ax.set_zorder(2)
            ax.imshow(_center_crop_square(img))
            label = original_sae_id_to_label.get(sae_id, f"ID {sae_id}")
            fontsize = max(6, min(10, int(image_size * 4)))
            ax.set_title(label, fontsize=fontsize, pad=2)
            ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Folder-structure export
# ---------------------------------------------------------------------------

def export_dag_as_folder_structure(G, root_node_id, topk_indices, dataset,
                                   original_sae_id_to_label, export_root,
                                   max_images_per_node=3, max_depth=None):
    """
    Export the subDAG as a nested folder hierarchy.

    Each graph node becomes a directory containing:
      - one cropped JPEG per associated SAE feature (named ``{sae_id}_{label}.jpg``)
      - ``node_info.json`` with node_id, level, and feature metadata
    Children of a node are nested as sub-directories inside their parent.

    DAG diamonds (nodes with multiple parents) are duplicated so the full
    hierarchy is self-contained in the folder tree.

    Args:
        export_root: Root directory for the export.  Created if absent.
    """
    from PIL import Image as PILImage

    all_nodes = {root_node_id} | nx.descendants(G, root_node_id)
    subG = G.subgraph(all_nodes).copy()
    levels = _compute_dag_levels(subG, root_node_id)

    if max_depth is not None:
        all_nodes = {n for n, lv in levels.items() if lv <= max_depth}
        subG = subG.subgraph(all_nodes).copy()
        levels = {n: lv for n, lv in levels.items() if n in all_nodes}

    def _safe(s):
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(s))

    def _save_img(img_arr, path):
        if img_arr.dtype != np.uint8:
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        pil = PILImage.fromarray(img_arr)
        ext = ".png" if img_arr.shape[-1] == 4 else ".jpg"
        path = os.path.splitext(path)[0] + ext
        pil.save(path, quality=95)
        return os.path.basename(path)

    def _export_node(node_id, node_dir):
        os.makedirs(node_dir, exist_ok=True)

        sae_ids = G.nodes[node_id].get('associated_original_sae_ids', [])[:max_images_per_node]
        features = []
        for sae_id in sae_ids:
            label = original_sae_id_to_label.get(sae_id, f"id_{sae_id}")
            img = dataset[topk_indices[sae_id][0]]
            if isinstance(img, tuple):
                img = img[0]
            img_arr = _center_crop_square(img)
            filename = _save_img(img_arr, os.path.join(node_dir, f"{sae_id}_{_safe(label)}"))
            features.append({"sae_id": int(sae_id), "label": label, "image": filename})

        with open(os.path.join(node_dir, "node_info.json"), "w") as f:
            json.dump({"node_id": int(node_id), "level": levels[node_id], "features": features}, f, indent=2)

        for child in subG.successors(node_id):
            _export_node(child, os.path.join(node_dir, f"node_{child}"))

    root_dir = os.path.join(export_root, f"node_{root_node_id}")
    _export_node(root_node_id, root_dir)
    print(f"Exported DAG folder structure to {root_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_start_str="msae_rw", root_node_id=3959, max_depth=None):
    config = None
    for config_name, cfg in CONFIGS.items():
        if config_name.startswith(config_start_str):
            config = cfg
            print(f"Selected config: {config_name}")
            break

    if config is None:
        raise ValueError(f"No config found starting with '{config_start_str}'")

    print("Loading graph...")
    G, _subset_map, sae_ids, labels = get_graph(config)
    original_sae_id_to_label = dict(zip(sae_ids, labels))

    print("Loading top-k activating images...")
    topk_indices, _ = SAEDimensions(config).get_topk_activating_images_indices_and_values()

    print("Loading dataset...")
    dataset = load_dataset(config.graph_eval_dataset, False)

    folder_name = "experiments_results/qualitative_evaluation/dag_visualizations"
    os.makedirs(folder_name, exist_ok=True)

    depth_suffix = f"_depth{max_depth}" if max_depth is not None else ""
    save_path = f"{folder_name}/{config_start_str}_dag_{root_node_id}{depth_suffix}.png"

    fig = create_dag_visualization(
        G=G,
        root_node_id=root_node_id,
        topk_indices=topk_indices,
        dataset=dataset,
        original_sae_id_to_label=original_sae_id_to_label,
        image_size=2.0,
        max_images_per_node=3,
        max_depth=max_depth,
        save_path=save_path,
    )

    export_dag_as_folder_structure(
        G=G,
        root_node_id=root_node_id,
        topk_indices=topk_indices,
        dataset=dataset,
        original_sae_id_to_label=original_sae_id_to_label,
        export_root=f"{folder_name}/{config_start_str}_dag_{root_node_id}{depth_suffix}",
        max_images_per_node=3,
        max_depth=max_depth,
    )

    plt.show()



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DAG Tree Visualization")
    parser.add_argument("--config_start_str", type=str, default="msae_rw",
                        help="Prefix of config name to select from CONFIGS")
    parser.add_argument("--root_node_id", '-r',  type=int, default=3959,
                        help="Root node ID for the subDAG visualization")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Max depth from root to include in visualization (None = full)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for config_start_str in CONFIGS.keys():
        if "hsae" not in config_start_str or "seed42" not in config_start_str:
            continue
        np.random.seed(1)
        main(config_start_str=config_start_str, root_node_id=args.root_node_id, max_depth=args.max_depth)

from typing import List
import pandas as pd
import networkx as nx

from my_config import DEFAULT_CONFIG
from structure_evaluation.clustering_graph_metrics.Intra_cluster_metric_runner import MetricRunner
from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import Metric
from structure_extraction.structure_extraction_utils import get_graph


class DepthMetricRunner(MetricRunner):
    """Computes per-cluster depth (shortest path from root) for a single config."""

    FILENAME = "depth_metrics.pkl"

    def __init__(self, metrics: List[Metric] = None):
        super().__init__(metrics or [])

    def compute_metrics_for_single_config(self, config_name: str, config) -> pd.DataFrame:
        all_results = []

        print(f"\nProcessing {config_name}...")

        G, subset_sae_id_to_original_sae_id_map, _, _ = get_graph(config)
        cluster_nodes_to_use = self._get_cluster_nodes(G)

        if len(cluster_nodes_to_use) == 0:
            return pd.DataFrame(all_results)

        root_node = [n for n, d in G.in_degree() if d == 0][0]
        shortest_depths = nx.shortest_path_length(G, source=root_node)

        for node in cluster_nodes_to_use:
            parent_sae_id, sae_ids_of_parent_subnodes = self._get_subnodes_of_parent(
                node, G, subset_sae_id_to_original_sae_id_map, mode=self.SUBNODE_MODE
            )

            all_results.append({
                'config_name': config_name,
                'node_id': node,
                'parent_sae_id': parent_sae_id,
                'num_children': len(sae_ids_of_parent_subnodes),
                'child_sae_ids': sae_ids_of_parent_subnodes,
                'depth_from_root_shortest': shortest_depths[node],
            })

        return pd.DataFrame(all_results)


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    runner = DepthMetricRunner(metrics=[])
    results_df = runner.compute_metrics_for_single_config(config.simple_name, config)
    DepthMetricRunner.save_results(results_df, config)

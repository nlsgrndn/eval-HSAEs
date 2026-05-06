from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import Metric, ClusterData
from typing import List
# from my_config import CONFIGS
from my_config import DEFAULT_CONFIG
from structure_evaluation.clustering_graph_metrics.Intra_cluster_metric_runner import MetricRunner
import pandas as pd
from structure_extraction.structure_extraction_utils import get_graph
import networkx as nx
from utils_sae_feature_properties import SAEDimensions
import numpy as np

class MetadataMetricRunner(MetricRunner):
    """Orchestrates metric computation across configs and clusters for activation-based metrics."""
    
    FILENAME = "metadata_metrics.pkl"
    
    def __init__(self, metrics: List[Metric]):
        super().__init__(metrics)

    def compute_metrics_for_single_config(self, config_name: str, config) -> pd.DataFrame:
        """Computes metrics for all clusters in a single config."""
        all_results = []
        
        print(f"\nProcessing {config_name}...")
        
        # Load graph and activations for this config
        G, subset_sae_id_to_original_sae_id_map, sae_ids, labels = get_graph(config)
        cluster_nodes_to_use = self._get_cluster_nodes(G)
        
        # early return if cluster_nodes_to_use is empty
        if len(cluster_nodes_to_use) == 0:
            return pd.DataFrame(all_results)

        # Initialize row with metadata
        row = {
            'config_name': config_name,
        }
        
        G_without_root = G.copy()
        root_node = [node for node in G_without_root.nodes if G_without_root.in_degree(node) == 0][0]
        G_without_root.remove_node(root_node)

        # add number of nodes in the graph
        # only consider nodes that have associated subset SAE IDs (i.e., nodes that are relevant for clustering)
        row['num_nodes_in_graph'] = len(G_without_root.nodes)
        
        # add number of edges in the graph
        row['num_edges_in_graph'] = len(G_without_root.edges)

        # add number of clusters
        row['num_clusters_in_graph'] = len(cluster_nodes_to_use)

        # add average cluster size
        cluster_sizes = [G_without_root.out_degree(node) for node in cluster_nodes_to_use]
        cluster_sizes_arr = np.array(cluster_sizes)
        row['avg_cluster_size'] = cluster_sizes_arr.mean()
        row['std_cluster_size'] = cluster_sizes_arr.std()



        # for all nodes with indegree > 0, compute the average indegree and add to row
        indegrees = [G_without_root.in_degree(node) for node in G_without_root.nodes if G_without_root.in_degree(node) > 0]
        row['avg_indegree'] = sum(indegrees) / len(indegrees) if indegrees else 0

        depth_dict_longest= {root_node: 0}
        for u in nx.topological_sort(G):
            if u not in depth_dict_longest:
                continue
            for v in G.successors(u):
                cand = depth_dict_longest[u] + 1
                if cand > depth_dict_longest.get(v, float("-inf")):
                    depth_dict_longest[v] = cand

        # number of nodes at depth at each depth level
        depth_counts = {}
        for depth in depth_dict_longest.values():
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1
        
        max_depth = max(depth_counts.keys())

        row['max_depth_longest_path'] = max_depth
        for depth in range(max_depth + 1):
            row[f'num_nodes_at_depth_{depth}_longest_path'] = depth_counts.get(depth, 0)
        



        # nodes at depth i ; note that graph is a DAG so we can compute depth using shortest path from root; use the original graph with root for this computation
        depth_dict_shortest = {}
        for node in G.nodes:
            depth_dict_shortest[node] = nx.shortest_path_length(G, source=root_node, target=node)
        # number of nodes at depth at each depth level
        depth_counts = {}
        for depth in depth_dict_shortest.values():
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1
        
        max_depth = max(depth_counts.keys())

        row['max_depth'] = max_depth

        for depth in range(max_depth + 1):
            row[f'num_nodes_at_depth_{depth}'] = depth_counts.get(depth)
        
        interpretabililty_df = SAEDimensions(config).get_interpretability_data()
        # get the column feature_density from interpretability_df for the sae_ids in the graph; then take the average and add to row
        original_sae_ids_in_graph = G.nodes()[root_node]['all_associated_original_sae_ids_in_subDAG']
        feature_density_values = interpretabililty_df.loc[interpretabililty_df['sae_id'].isin(original_sae_ids_in_graph), 'feature_density'].values
        row['avg_feature_density'] = feature_density_values.mean()

        all_results.append(row)
        
        return pd.DataFrame(all_results)
    

if __name__ == "__main__":
    # Example usage
    metrics_to_run = []

    config = DEFAULT_CONFIG
    runner = MetadataMetricRunner(metrics=metrics_to_run)
    results_df = runner.compute_metrics_for_single_config(config.simple_name, config)
    MetadataMetricRunner.save_results(results_df, config)


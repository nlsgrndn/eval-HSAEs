from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import Metric, ClusterData
from typing import List
# from my_config import CONFIGS
from my_config import DEFAULT_CONFIG
from structure_evaluation.clustering_graph_metrics.Intra_cluster_metric_runner import MetricRunner
import pandas as pd
from structure_extraction.structure_extraction_utils import get_graph
from utils_sae_feature_properties import SAEDimensions
import networkx as nx

class FeatureLevelMetricRunner(MetricRunner):
    """Orchestrates metric computation across configs and clusters for activation-based metrics."""
    
    FILENAME = "feature_level_metrics.pkl"
    
    def __init__(self, metrics: List[Metric]):
        assert len(metrics) == 0, "For feature-level metrics, we are specifying the metrics via constant. Please initialize with an empty list of metrics."
        super().__init__(None)

    def compute_metrics_for_single_config(self, config_name: str, config) -> pd.DataFrame:
        """Computes metrics for all clusters in a single config."""
        all_results = []
        
        print(f"\nProcessing {config_name}...")
        
        # Load graph and activations for this config
        G, subset_sae_id_to_original_sae_id_map, sae_ids, labels = get_graph(config)
        interpretabililty_df = SAEDimensions(config).get_interpretability_data()
        
          # Debugging breakpoint

        # early return if cluster_nodes_to_use is empty
        if len(G.nodes()) <= 1:
            return pd.DataFrame(all_results)
        
        metric_names = ["clarity_score", "ms_score"]  # These are the columns in interpretability_df we want to extract
        for node_id, node_data in G.nodes(data=True):
            # skip artificial root node if it exists
            if G.in_degree(node_id) == 0:
                continue

            # get original sae_ids corresponding to this node
            subset_sae_id = node_data["associated_sae_subset_ids"][0]
            original_sae_id = node_data["associated_original_sae_ids"][0]
            # assert that same as with mapping
            assert original_sae_id == subset_sae_id_to_original_sae_id_map[subset_sae_id], f"Mismatch for node {node_id}: {original_sae_id} vs {subset_sae_id_to_original_sae_id_map[subset_sae_id]}"

            # Initialize row with metadata
            row = {
                'config_name': config_name,
                'node': node_id,
                'subset_sae_id': subset_sae_id,
                'original_sae_id': original_sae_id
            }

            for metric in metric_names:
                metric_results = interpretabililty_df.loc[interpretabililty_df['sae_id'] == original_sae_id, metric].values
                metric_results = metric_results[0] if len(metric_results) > 0 else None
                row.update({metric: metric_results})
            all_results.append(row)
        
        return pd.DataFrame(all_results)
    

if __name__ == "__main__":
    # Example usage
    metrics_to_run = []

    config = DEFAULT_CONFIG
    runner = FeatureLevelMetricRunner(metrics=metrics_to_run)
    results_df = runner.compute_metrics_for_single_config(config.simple_name, config)
    FeatureLevelMetricRunner.save_results(results_df, config)


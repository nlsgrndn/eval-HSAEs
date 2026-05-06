import numpy as np
import pandas as pd
from structure_evaluation.clustering_graph_metrics.intra_cluster_metrics_utils import ClusterData, Metric
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from path_hub import PathBuilder
import gc

from structure_extraction.structure_extraction_utils import get_graph
import os

class MetricRunner:
    """Orchestrates metric computation across clusters for a single config."""
    
    FILENAME = None # Placeholder, to be defined in subclasses
    NUM_CHILDREN_THRESHOLD = 0  # Minimum number of children to consider a cluster
    SUBNODE_MODE = "children"  # Mode to get subnodes: "children" or "all_descendants"

    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
    
    def compute_metrics_for_single_config(self, config_name: str, config) -> pd.DataFrame:
        """Computes metrics for all clusters in a single config."""
        all_results = []
        
        print(f"\nProcessing {config_name}...")
        
        # Load graph and activations for this config
        G, subset_sae_id_to_original_sae_id_map, sae_ids, labels = get_graph(config)
        cluster_nodes_to_use = self._get_cluster_nodes(G)
        typed_dict = self._get_model_level_data(config)
        
        # Process each cluster
        for node in tqdm(cluster_nodes_to_use, desc=f"Clusters in {config_name}", leave=False):
            # Get subnodes of parent
            parent_sae_id, sae_ids_of_parent_subnodes = self._get_subnodes_of_parent(
                node, G, subset_sae_id_to_original_sae_id_map, mode=self.SUBNODE_MODE
            )
            
            # Create cluster data
            cluster_data = self._create_cluster_data(
                config_name, node, parent_sae_id, sae_ids_of_parent_subnodes, typed_dict=typed_dict
            )
            
            # Apply common filtering (independent of specific metrics)
            if self._filter_cluster(cluster_data):
                continue
            
            # Initialize row with metadata
            row = {
                'config_name': config_name,
                'node_id': node,
                'parent_sae_id': parent_sae_id,
                'num_children': cluster_data.num_children,
                'child_sae_ids': cluster_data.child_sae_ids
            }
            
            # Compute each metric and merge results into row
            for metric in self.metrics:
                metric_results = metric.compute(cluster_data)
                row.update(metric_results)
            
            all_results.append(row)

        # Free memory after processing this config
        del typed_dict
        gc.collect()  # Force garbage collection to free memory immediately
        
        return pd.DataFrame(all_results)
    
    @staticmethod
    def _get_subnodes_of_parent(node, G, subset_sae_id_to_original_sae_id_map, mode):
        node_data = G.nodes[node]
        parent_sae_id = subset_sae_id_to_original_sae_id_map[node_data["associated_sae_subset_ids"][0]]
        
        if mode == "all_descendants":
            # get all descendants in the subDAG
            sae_ids_of_parent_subnodes = node_data["all_associated_original_sae_ids_in_subDAG"]
        elif mode == "children":
            # get successors aka children only
            sae_ids_of_parent_subnodes = [subset_sae_id_to_original_sae_id_map[G.nodes[child_node_id]["associated_sae_subset_ids"][0]] for child_node_id in G.successors(node)]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return parent_sae_id, sae_ids_of_parent_subnodes
    
    @classmethod
    def save_results(cls, df: pd.DataFrame, config) -> None:
        """Saves the results DataFrame to a pickle file."""
        if cls.FILENAME is None:
            raise ValueError("FILENAME class attribute must be defined")
        
        output_path = os.path.join( 
            PathBuilder(config=config).get_hierarchical_graph_eval_path(),
            cls.FILENAME
        )
        df.to_pickle(output_path)
        
        print(f"\nSaved results to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")

    @classmethod
    def load_results(cls, config) -> pd.DataFrame:
        """Loads the results DataFrame from a pickle file."""
        if cls.FILENAME is None:
            raise ValueError("FILENAME class attribute must be defined")
        
        input_path = os.path.join( 
            PathBuilder(config=config).get_hierarchical_graph_eval_path(),
            cls.FILENAME
        )
        df = pd.read_pickle(input_path)
        
        print(f"\nLoaded results from {input_path}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        return df

    
    def _get_cluster_nodes(self, G) -> List[int]:
        return [node for node in G.nodes() if G.out_degree(node) != 0 and G.in_degree(node) > 0]
    
    # def _get_cluster_nodes(self, G):
    #     root_node = [n for n, d in G.in_degree() if d == 0][0]
    #     return list(G.successors(root_node))

    def _get_model_level_data(self, config):
        # Placeholder for method to get model-level data
        pass

    def _filter_cluster(self, cluster_data: ClusterData) -> bool:
        """Filter clusters based on minimum size."""
        return False
        # # Minimum cluster size
        # return cluster_data.num_children < self.NUM_CHILDREN_THRESHOLD

    def _create_cluster_data(self, config_name: str, node: Any, parent_sae_id: int,
                            sae_ids_of_parent_subnodes: List[int], typed_dict = None) -> ClusterData:
        """Extract and prepare cluster data."""
        pass

    
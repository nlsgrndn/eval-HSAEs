import pandas as pd
import os
import networkx as nx
import pickle
from label_assignment_strategies.load_sae_labelling_and_metrics_results import load_sae_ids_with_labels
from path_hub import PathBuilder
import random

def get_graph(sae_config):
    """Load graph structure for a config."""
    selected_path_builder = PathBuilder(config=sae_config)
    sae_ids, labels = load_sae_ids_with_labels(config=sae_config)
    graph_name = sae_config.graph_name
    graph_path = os.path.join(
        selected_path_builder.get_hierarchical_graphs_path(),
        f"{graph_name}.pkl"
    )
    reader = NetworkXClusteringResultReader(graph_path, original_sae_ids=sae_ids, labels=labels)
    G = reader.result_graph
    subset_sae_id_to_original_sae_id_map = {i: sae_id for i, sae_id in enumerate(sae_ids)}
    return G, subset_sae_id_to_original_sae_id_map, sae_ids, labels

def save_clustering_results_df(df: pd.DataFrame, clustering_approach_name):
    os.makedirs(PathBuilder().get_cluster_labels_path(), exist_ok=True)
    path = os.path.join(PathBuilder().get_cluster_labels_path(), f"{clustering_approach_name}_labels.csv")
    df.to_csv(path, index=False)

def save_clustering_results(sae_id_list, labels, clustering_approach_name):
    df = pd.DataFrame({"sae_id": sae_id_list, "cluster_label": labels})
    save_clustering_results_df(df, clustering_approach_name)

def load_clustering_results(clustering_approach_name):
    path = os.path.join(PathBuilder().get_cluster_labels_path(), f"{clustering_approach_name}_labels.csv")
    df = pd.read_csv(path)
    sae_ids = df["sae_id"].tolist()
    cluster_labels = df["cluster_label"].fillna("-1").astype(str).tolist()
    return sae_ids, cluster_labels

class NetworkXClusteringResultReader:
    """Reads a clustering result stored as a pickled NetworkX DiGraph.
    
    Input assumptions:
    - The pickled file contains a NetworkX DiGraph.
    - Expects each node to have an attribute "associated_sae_subset_ids" which is a list of SAE subset IDs.
    - Node ids are assumed to be integers.
    
    The class augments the graph by adding:
    - "all_associated_sae_subset_ids_in_subDAG": list of all SAE subset IDs in the subtree rooted at that node.
    - "all_associated_labels_in_subDAG": list of all labels in the subtree (if labels are provided).
    - "associated_labels": list of labels directly associated with the node (if labels are provided).
    - "associated_original_sae_ids": list of original SAE IDs directly associated with the node (if original SAE IDs are provided).
    - "node_name": a human-readable name for the node, if a mapping is provided.

    """

    def __init__(self, pkl_file_path, original_sae_ids = None, labels = None):
        self.pkl_file_path = pkl_file_path
        self.original_sae_ids = original_sae_ids
        self.labels = labels
        self.sae_subset_id_to_label_map = dict(zip(list(range(len(labels))), labels)) if labels is not None else None
        self.sae_subset_id_to_original_id_map = dict(zip(list(range(len(original_sae_ids))), original_sae_ids)) if original_sae_ids is not None else None
        self.result_graph = self._load_graph_from_pickle()
        self._check_graph_properties()
        self._augment_with_node_name()
        self._augment_with_associated_sae_subset_ids_subtree()
        # self._add_depth_attribute_to_graph()
        self._augment_networkX_graph_attributes()

    def _load_graph_from_pickle(self):
        with open(self.pkl_file_path, "rb") as f:
            G = pickle.load(f)
        return G
    
    def _check_graph_properties(self):
        G = self.result_graph
        # DAG check
        is_dag = nx.is_directed_acyclic_graph(G)
        # check that all nodes have associated_sae_subset_ids attribute and it contains a list of at most one id
        all_have_associated_ids = all(
            "associated_sae_subset_ids" in G.nodes[node] and
            isinstance(G.nodes[node]["associated_sae_subset_ids"], list) and
            len(G.nodes[node]["associated_sae_subset_ids"]) <= 1
            for node in G.nodes()
        )
        # check that one sae subset id is associated to at most one node
        sae_id_to_node = {}
        all_have_unique_association = True
        for node in G.nodes():
            for sae_id in G.nodes[node]["associated_sae_subset_ids"]:
                if sae_id in sae_id_to_node:
                    all_have_unique_association = False
                    break
                sae_id_to_node[sae_id] = node
            if not all_have_unique_association:
                break

        assert is_dag, f"Graph at {self.pkl_file_path} is not a DAG"
        assert all_have_associated_ids, f"Not all nodes in graph at {self.pkl_file_path} have valid 'associated_sae_subset_ids' attribute"
        assert all_have_unique_association, f"Some SAE subset IDs are associated to multiple nodes in graph at {self.pkl_file_path}"

    def _augment_with_associated_sae_subset_ids_subtree(self):
        # For each node, find all associated SAE IDs in its subtree
        for node in self.result_graph.nodes():
            node_id = node
            data = self.result_graph.nodes[node]
            # Find all descendants
            descendants = nx.descendants(self.result_graph, node_id)
            # Get all associated SAE IDs from descendants
            all_associated_sae_subset_ids = []
            for desc in descendants:
                all_associated_sae_subset_ids.extend(self.result_graph.nodes[desc].get("associated_sae_subset_ids", []))
            # Include the node's own associated SAE IDs
            all_associated_sae_subset_ids.extend(data.get("associated_sae_subset_ids", []))
            data["all_associated_sae_subset_ids_in_subDAG"] = all_associated_sae_subset_ids

            # check for duplicates
            if len(all_associated_sae_subset_ids) != len(set(all_associated_sae_subset_ids)):
                raise ValueError(f"Node {node_id} has duplicate SAE subset IDs in its subtree.")

    def _augment_with_node_name(self):
        # For each node, find its name
        if self.sae_subset_id_to_label_map is None:
            return

        for node in self.result_graph.nodes():
            data = self.result_graph.nodes[node]
            # Check if node_name already exists
            if "node_name" in data:
                continue
            associated_ids = data.get("associated_sae_subset_ids", [])
            if len(associated_ids) == 1:
                sae_subset_id = associated_ids[0]
                sae_subset_id_label = self.sae_subset_id_to_label_map.get(sae_subset_id)
                if sae_subset_id_label:
                    data["node_name"] = sae_subset_id_label
            elif len(associated_ids) > 1:
                data["node_name"] = "n_>1"
            else:
                data["node_name"] = "n_0"


    def _augment_networkX_graph_attributes(self):
        if self.sae_subset_id_to_label_map is None and self.sae_subset_id_to_original_id_map is None:
            return
        G = self.result_graph

        for node in self.result_graph.nodes(data=True):
            if self.sae_subset_id_to_label_map is not None:
                G.nodes[node[0]]["all_associated_labels_in_subDAG"] = [self.sae_subset_id_to_label_map[int(f)] for f in G.nodes[node[0]]["all_associated_sae_subset_ids_in_subDAG"]]
                G.nodes[node[0]]["associated_labels"] = [self.sae_subset_id_to_label_map[int(f)] for f in G.nodes[node[0]]["associated_sae_subset_ids"]]
            if self.sae_subset_id_to_original_id_map is not None:
                G.nodes[node[0]]["all_associated_original_sae_ids_in_subDAG"] = [self.sae_subset_id_to_original_id_map[int(f)] for f in G.nodes[node[0]]["all_associated_sae_subset_ids_in_subDAG"]]
                G.nodes[node[0]]["associated_original_sae_ids"] = [self.sae_subset_id_to_original_id_map[int(f)] for f in G.nodes[node[0]]["associated_sae_subset_ids"]]

import pickle
class NetworkXClusteringResultSaver():

    @staticmethod
    def save_conditional_activation_hierarchy_graph(G: nx.DiGraph, name):
        output_file_path = os.path.join(PathBuilder().get_hierarchical_graphs_path(), f"{name}.pkl")
        with open(output_file_path, "wb") as f:
            pickle.dump(G, f)
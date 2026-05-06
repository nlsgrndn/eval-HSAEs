import networkx as nx
import numpy as np
from path_hub import PathBuilder
from activations_preprocessing.post_process_cond_acts import filter_matrix, filter_original_sae_ids
from structure_extraction.structure_extraction_utils import NetworkXClusteringResultSaver
from label_assignment_strategies.load_sae_labelling_and_metrics_results import load_sae_ids_with_labels
from activations_preprocessing.dependency_metrics import SaveAndLoad
from activations_preprocessing.masked_cosine_similarity import load_mcs_matrix
from configs.mcs_graph_config import MCSGraphConfig
from configs.conditional_act_graph_config import ConditionalActivationGraphConfig
from configs.weighted_conditional_act_graph_config import WeightedConditionalActivationGraphConfig
from configs.graph_configs import get_graph_config
from my_config import DEFAULT_CONFIG

def get_edges_with_weights_for_mcs_activation_matrix(mcs_matrix, config: MCSGraphConfig):
    # Add edges where (mcs_matrix[i, j] > MCS_PARENT_GIVEN_CHILD_THRESHOLD) &  (mcs_matrix[j, i] < MCS_CHILD_GIVEN_PARENT_THRESHOLD) using vectorized operations
    # M_ij contains masked cosine similarity between features i and j, with masking on i being active
    # condition 1: MCS(child, parent) > MCS_PARENT_GIVEN_CHILD_THRESHOLD
    # condition 2: MCS(parent, child) < MCS_CHILD_GIVEN_PARENT_THRESHOLD (ensure that the relationship is not bidirectional)
    condition_1 = mcs_matrix > config.mcs_parent_given_child_threshold
    condition_2 = mcs_matrix.T < config.mcs_child_given_parent_threshold
    child_indices, parent_indices = np.where(condition_1 & condition_2)
    edge_weights = mcs_matrix[child_indices, parent_indices]
    edges_with_weights = [(int(p), int(c), {'weight': float(w)})
                        for p, c, w in zip(parent_indices, child_indices, edge_weights)]
    return edges_with_weights

def get_edges_with_weights_for_conditional_activation_matrix(cond_act_matrix, config: ConditionalActivationGraphConfig):
    # Add edges where (cond_act_matrix[i, j] > COND_ACT_PARENT_GIVEN_CHILD_THRESHOLD) &  (cond_act_matrix[j, i] < COND_ACT_CHILD_GIVEN_PARENT_THRESHOLD) & using vectorized operations
    # M_ij contains P(j active | i active)
    # condition 1: P(parent | child) > COND_ACT_PARENT_GIVEN_CHILD_THRESHOLD
    # condition 2: P(child | parent) < COND_ACT_CHILD_GIVEN_PARENT_THRESHOLD (ensure that the relationship is not bidirectional)
    condition_1 = cond_act_matrix > config.conditional_activation_parent_given_child_threshold
    condition_2 = cond_act_matrix.T < config.conditional_activation_child_given_parent_threshold
    child_indices, parent_indices = np.where(condition_1 & condition_2)
    edge_weights = cond_act_matrix[child_indices, parent_indices]
    edges_with_weights = [(int(p), int(c), {'weight': float(w)})
                        for p, c, w in zip(parent_indices, child_indices, edge_weights)]
    return edges_with_weights

def get_edges_with_weights_for_weighted_conditional_activation_matrix(weighted_condact_matrix, config: WeightedConditionalActivationGraphConfig):
    # Add edges where (weighted_condact_matrix[i, j] > COND_ACT_PARENT_GIVEN_CHILD_THRESHOLD) &  (weighted_condact_matrix[j, i] < COND_ACT_CHILD_GIVEN_PARENT_THRESHOLD) & using vectorized operations
    # M_ij contains P_weighted(j active | i active)
    # condition 1: P_weighted(parent | child) > COND_ACT_PARENT_GIVEN_CHILD_THRESHOLD
    # condition 2: P_weighted(child | parent) < COND_ACT_CHILD_GIVEN_PARENT_THRESHOLD (ensure that the relationship is not bidirectional)
    condition_1 = weighted_condact_matrix > config.conditional_activation_parent_given_child_threshold
    condition_2 = weighted_condact_matrix.T < config.conditional_activation_child_given_parent_threshold
    child_indices, parent_indices = np.where(condition_1 & condition_2)
    edge_weights = weighted_condact_matrix[child_indices, parent_indices]
    edges_with_weights = [(int(p), int(c), {'weight': float(w)})
                        for p, c, w in zip(parent_indices, child_indices, edge_weights)]
    return edges_with_weights


def temp_get_edges_with_weights_HSAE(original_sae_id_to_sae_subset_id_filtered):
    parents = 361
    children_for_parents = 16
    edges_with_weights = []
    for p in range(parents):
        for c in range(parents + p*children_for_parents, parents + (p+1)*children_for_parents):
            if p in original_sae_id_to_sae_subset_id_filtered and c in original_sae_id_to_sae_subset_id_filtered:
                edges_with_weights.append((original_sae_id_to_sae_subset_id_filtered.get(p), original_sae_id_to_sae_subset_id_filtered.get(c), {'weight': 1.0}))
    return edges_with_weights

def generate_graph_from_edge_weights(edges_with_weights, graph_config, sae_width, as_tree: bool = True, ):
    # Create a directed graph from the directionality matrix
    G_dir = nx.DiGraph()
    # filter the edges to only such that if a child_index occurs multiple times, only the one with the highest weight is kept
    # keep only the strongest incoming edge per child (if a child has multiple parents, keep the parent with highest weight)
    if as_tree:
        child_best = {}
        for p, c, attr in edges_with_weights:
            w = attr['weight']
            existing = child_best.get(c)
            if existing is None or w > existing[2]['weight']:
                child_best[c] = (int(p), int(c), {'weight': w})
        edges_with_weights = list(child_best.values())
    G_dir.add_edges_from(edges_with_weights)
    no_incoming = [node for node in G_dir.nodes() if G_dir.in_degree(node) == 0]
    # check that G_dir is DAG
    assert nx.is_directed_acyclic_graph(G_dir), "Generated graph is not a DAG"

    # Apply transitive reduction to remove redundant edges
    # This removes edges A->B if there exists a path A->X->B (or longer)
    # keeping only the direct hierarchical relationships
    edges_before = G_dir.number_of_edges()
    G_dir = nx.transitive_reduction(G_dir)
    edges_after = G_dir.number_of_edges()
    edges_removed = edges_before - edges_after
    print(f"Transitive reduction: removed {edges_removed} edges ({edges_before} -> {edges_after})")


    # remove parents with too many children (more than max_children)
    # NOTE: requires that isolated nodes after this step are removed in a subsequent step, since they would be uninformative to keep in the graph
    for node in list(G_dir.nodes()):
        children = list(G_dir.successors(node))
        if len(children) > graph_config.max_pct_children_rel_to_SAEwidth * sae_width:
            G_dir.remove_node(node)

    # Remove subdags with fewer than min_descendants nodes
    # Get all nodes with no incoming edges (entry nodes)
    no_incoming = [node for node in G_dir.nodes() if G_dir.in_degree(node) == 0]
    
    # First pass: mark all nodes reachable from entry nodes with sufficient subtree size
    nodes_to_keep = set()
    for root in no_incoming:
        num_descendants = len(nx.descendants(G_dir, root))
        if num_descendants >= graph_config.min_descendants:
            nodes_to_keep.update({root} | nx.descendants(G_dir, root))
    
    # Second pass: remove all unmarked nodes
    nodes_to_remove = set(G_dir.nodes()) - nodes_to_keep
    G_dir.remove_nodes_from(nodes_to_remove)

    # for each node add the attribute "associated_sae_subset_ids"
    for node in G_dir.nodes():
        G_dir.nodes()[node]["associated_sae_subset_ids"] = [node]

    # get all features with no incoming edges
    no_incoming = [node for node in G_dir.nodes() if G_dir.in_degree(node) == 0]

    # Add a dummy parent node that connects to all features with no incoming edges
    if len(G_dir.nodes()) == 0:
        max_node_id = -1
        print("WARNING: EMPTY GRAPH CREATED")
    else:
        max_node_id = max(G_dir.nodes())
    dummy_parent = max_node_id + 1
    G_dir.add_node(dummy_parent)
    G_dir.nodes()[dummy_parent]["associated_sae_subset_ids"] = []

    # connect dummy parent to all no incoming features
    for feature in no_incoming:
        G_dir.add_edge(dummy_parent, feature, weight=1.0)
    return G_dir

def reorder_by_subset_ids(matrix, sae_ids):
    subset_indices = np.array(sae_ids)
    reordered_matrix = matrix[np.ix_(subset_indices, subset_indices)]
    return reordered_matrix

def main(config):
    graph_config = get_graph_config()
    assert graph_config.name == config.graph_name, f"Graph config name {graph_config.name} does not match expected config name {config.name}"

    sae_width = config.sae_model.width

    if "hsae" in config.sae_model.name:
        dependency_metrics = SaveAndLoad.load_data_from_npy_arrays(
            cond_acts_metric_cfg=graph_config.metrics_config,
            path_builder=PathBuilder(),
        )
        base_rates = dependency_metrics['base_rates']
        original_sae_ids_to_exclude = filter_original_sae_ids(base_rates, **graph_config.filtering_kwargs)
        sae_ids, labels = load_sae_ids_with_labels()
        original_sae_id_to_sae_subset_id = {original_sae_id: subset_sae_id for subset_sae_id, original_sae_id in enumerate(sae_ids)}
        original_sae_id_to_sae_subset_id_filtered = {original_sae_id: subset_sae_id for original_sae_id, subset_sae_id in original_sae_id_to_sae_subset_id.items() if original_sae_id not in original_sae_ids_to_exclude}
        edges_with_weights = temp_get_edges_with_weights_HSAE(original_sae_id_to_sae_subset_id_filtered)
        G_dir = generate_graph_from_edge_weights(edges_with_weights, graph_config, sae_width, as_tree=(graph_config.structure_type=="tree"))
        graph_name = f"{graph_config.name}"
        print(f"Saving graph {graph_name} with {G_dir.number_of_nodes()} nodes and {G_dir.number_of_edges()} edges")
        NetworkXClusteringResultSaver.save_conditional_activation_hierarchy_graph(G_dir, graph_name)
    elif isinstance(graph_config, MCSGraphConfig):
        mcs_matrix, base_rates = load_mcs_matrix(graph_config.metrics_config)

        # set diagonal to 0
        np.fill_diagonal(mcs_matrix, 0.0)

        # replace nan values with 0
        mcs_matrix = np.nan_to_num(mcs_matrix, nan=0.0)
        base_rates = np.nan_to_num(base_rates, nan=0.0)

        # assert that no nan values remain
        assert not np.isnan(mcs_matrix).any(), "MCS matrix contains NaN values after replacement."
        assert not np.isnan(base_rates).any(), "Base rates contain NaN values after replacement."

        filtered_mcs_matrix = filter_matrix(mcs_matrix, base_rates, **graph_config.filtering_kwargs)

        sae_ids, labels = load_sae_ids_with_labels()
        # IMPORTANT: reorder the filtered mcs matrix indexed by sae subset ids. This is the expected format for graph creation.
        filtered_mcs_matrix = reorder_by_subset_ids(filtered_mcs_matrix, sae_ids)

        edges_with_weights = get_edges_with_weights_for_mcs_activation_matrix(filtered_mcs_matrix, graph_config)
        G_dir = generate_graph_from_edge_weights(edges_with_weights, graph_config, sae_width, as_tree=(graph_config.structure_type=="tree"))
        graph_name = f"{graph_config.name}"
        print(f"Saving graph {graph_name} with {G_dir.number_of_nodes()} nodes and {G_dir.number_of_edges()} edges")
        NetworkXClusteringResultSaver.save_conditional_activation_hierarchy_graph(G_dir, graph_name)
    elif isinstance(graph_config, ConditionalActivationGraphConfig):
        dependency_metrics = SaveAndLoad.load_data_from_npy_arrays(
            cond_acts_metric_cfg=graph_config.metrics_config,
            path_builder=PathBuilder(),
        )
        sae_ids, labels = load_sae_ids_with_labels()
        cond_act_matrix = dependency_metrics['P_j_given_i']
        base_rates = dependency_metrics['base_rates']
        filtered_cond_act_matrix = filter_matrix(cond_act_matrix, base_rates, **graph_config.filtering_kwargs)
        # IMPORTANT: reorder the filtered conditional activation matrix indexed by sae subset ids. This is the expected format for graph creation.
        filtered_cond_act_matrix = reorder_by_subset_ids(filtered_cond_act_matrix, sae_ids)
        edges_with_weights = get_edges_with_weights_for_conditional_activation_matrix(filtered_cond_act_matrix, graph_config)
        G_dir = generate_graph_from_edge_weights(edges_with_weights, graph_config, sae_width, as_tree=(graph_config.structure_type=="tree"))
        graph_name = f"{graph_config.name}"
        print(f"Saving graph {graph_name} with {G_dir.number_of_nodes()} nodes and {G_dir.number_of_edges()} edges")
        NetworkXClusteringResultSaver.save_conditional_activation_hierarchy_graph(G_dir, graph_name)
    elif isinstance(graph_config, WeightedConditionalActivationGraphConfig):
        dependency_metrics = SaveAndLoad.load_data_from_npy_arrays(
            cond_acts_metric_cfg=graph_config.metrics_config,
            path_builder=PathBuilder(),
        )
        sae_ids, labels = load_sae_ids_with_labels()
        weighted_condact_matrix = dependency_metrics['P_j_given_i_weighted']
        base_rates = dependency_metrics['base_rates']
        filtered_weighted_condact_matrix = filter_matrix(weighted_condact_matrix, base_rates, **graph_config.filtering_kwargs)
        # IMPORTANT: reorder the filtered weighted conditional activation matrix indexed by sae subset ids. This is the expected format for graph creation.
        filtered_weighted_condact_matrix = reorder_by_subset_ids(filtered_weighted_condact_matrix, sae_ids)
        edges_with_weights = get_edges_with_weights_for_weighted_conditional_activation_matrix(filtered_weighted_condact_matrix, graph_config)
        G_dir = generate_graph_from_edge_weights(edges_with_weights, graph_config, sae_width, as_tree=(graph_config.structure_type=="tree"))
        graph_name = f"{graph_config.name}"
        print(f"Saving graph {graph_name} with {G_dir.number_of_nodes()} nodes and {G_dir.number_of_edges()} edges")
        NetworkXClusteringResultSaver.save_conditional_activation_hierarchy_graph(G_dir, graph_name)
    else:
        raise ValueError(f"Unsupported graph config type: {type(graph_config)}")

if __name__ == "__main__":
    main(DEFAULT_CONFIG)
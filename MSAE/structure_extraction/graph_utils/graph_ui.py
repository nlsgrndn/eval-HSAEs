"""
Graph UI for visualizing hierarchical clustering results of SAE features.

This module provides a Dash-based interactive visualization for exploring
hierarchical graph structures generated from SAE feature coactivation analysis.
"""

import base64
import io
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import dash_cytoscape as cyto
import matplotlib.pyplot as plt
import networkx as nx
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
from label_assignment_strategies.load_sae_labelling_and_metrics_results import load_sae_ids_with_labels
from my_utils import load_dataset
from structure_extraction.structure_extraction_utils import (
    NetworkXClusteringResultReader,
)
from activations_preprocessing.utils_sae_activations import get_precomputed_top_k_path, load_precomputed_top_k
from utils_figure_generation import create_top_activating_image_grid
from my_config import DEFAULT_CONFIG, Dataset
import random
from path_hub import PathBuilder

# Configuration constants
GRAPH_UI_ASSETS_DIR = "graph_ui_assets"
MAX_LABELS_DISPLAY = 20
MAX_REPRESENTATIVE_LABELS = 20
SINGLE_IMAGE_DISPLAY = 2

# Feature-image display: flip USE_IMAGE_GRID to False to revert to single top image.
USE_IMAGE_GRID = False
GRID_ROWS, GRID_COLS = 2, 2
TOP_K_IMAGES_PER_FEATURE = GRID_ROWS * GRID_COLS if USE_IMAGE_GRID else 1
MAX_IMAGES_DISPLAY = 9 if USE_IMAGE_GRID else 20
IMAGE_DISPLAY_WIDTH_PX = 180 if USE_IMAGE_GRID else 60
# Grid mode preserves aspect ratio; single-image mode forces a square thumbnail.
IMAGE_DISPLAY_HEIGHT = 'auto' if USE_IMAGE_GRID else f'{IMAGE_DISPLAY_WIDTH_PX}px'

HIERARCHY_GRAPHS_PATH = PathBuilder().get_hierarchical_graphs_path()


class DataManager:
    """Handles data loading and caching for the graph UI."""
    
    def __init__(self, inference_dataset: Dataset, assets_dir: str = GRAPH_UI_ASSETS_DIR):
        self.assets_dir = assets_dir
        self.graph_eval_dataset = inference_dataset
        self.dataset = load_dataset(inference_dataset, return_file_path=True)
        self._pil_dataset = load_dataset(inference_dataset, return_file_path=False)
        topk_path = get_precomputed_top_k_path()
        self._topk_indices, self._topk_values, _ = load_precomputed_top_k(topk_path)
        self._grid_uri_cache: Dict[int, str] = {}

    def load_graph(self, path: str) -> nx.DiGraph:
        """Load graph from file with proper error handling."""
        try:
            sae_ids, sae_labels = load_sae_ids_with_labels()
            reader = NetworkXClusteringResultReader(
                path,
                original_sae_ids=sae_ids,
                labels=sae_labels
            )
            return reader.result_graph
        except Exception as e:
            raise RuntimeError(f"Failed to load graph from {path}: {e}")

    def get_image_grid_data_uri(self, sae_id: int) -> Optional[str]:
        """Render a grid of the top activating images for a feature as a base64 data URI."""
        if sae_id in self._grid_uri_cache:
            return self._grid_uri_cache[sae_id]
        try:
            indices = self._topk_indices[sae_id, :TOP_K_IMAGES_PER_FEATURE]
            values = self._topk_values[sae_id, :TOP_K_IMAGES_PER_FEATURE]
            fig = create_top_activating_image_grid(
                activations=None,
                sae_id=sae_id,
                dataset=self._pil_dataset,
                num_rows=GRID_ROWS,
                num_cols=GRID_COLS,
                mode="return",
                top_images=(indices, values),
            )
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            self._grid_uri_cache[sae_id] = uri
            return uri
        except Exception as e:
            print(f"Warning: Failed to render grid for SAE ID {sae_id}: {e}")
            return None

    def get_image_data_uri(self, sae_id: int) -> Optional[str]:
        """Get the displayable image data URI for a feature.

        When USE_IMAGE_GRID is True, returns a base64 PNG of the top-k grid
        (falling back to the single top image on render failure).
        When False, returns the raw single top activating image directly.
        """
        if USE_IMAGE_GRID:
            uri = self.get_image_grid_data_uri(sae_id)
            if uri is not None:
                return uri
        single_path = self._get_single_top_activating_image_path(sae_id)
        if single_path is not None:
            return self._encode_image_file_to_data_uri(single_path)
        return None

    def _encode_image_file_to_data_uri(self, image_path: str) -> Optional[str]:
        """Convert image file to base64 data URI."""
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{encoded}"
        except (FileNotFoundError, IOError):
            return None

    def save_top_activating_image_paths(self) -> None:
        """Save top activating image paths to file (single-image fallback)."""
        try:
            precomputed_topk_path = get_precomputed_top_k_path()
            os.makedirs(self.assets_dir, exist_ok=True)
            assets_file_path = self._create_assets_filepath(precomputed_topk_path)
            top_activating_image_per_sae_id = self._topk_indices[:, 0]
            top_image_paths = [self.dataset[id][0] for id in top_activating_image_per_sae_id]

            with open(assets_file_path, "w") as f:
                for path in top_image_paths:
                    f.write(f"{path}\n")
        except Exception as e:
            print(f"Warning: Failed to save image paths: {e}")

    def _get_single_top_activating_image_path(self, index: int) -> Optional[str]:
        """Get top activating image path by index (single-image fallback)."""
        try:
            precomputed_topk_path = get_precomputed_top_k_path()
            assets_file_path = self._create_assets_filepath(precomputed_topk_path)
            with open(assets_file_path, "r") as f:
                lines = f.readlines()
                if 0 <= index < len(lines):
                    return lines[index].strip()
        except (FileNotFoundError, IOError):
            pass
        return None

    def _create_assets_filepath(self, precomputed_topk_path: str) -> str:
        """Create a safe filename for assets based on original path."""
        base_name = os.path.basename(precomputed_topk_path).rsplit(".", 1)[0]
        return os.path.join(self.assets_dir, f"{base_name}_top_activating_image_paths.txt")



class GraphUI:
    """Handles all UI-related functionality for the graph visualization."""
    
    def __init__(self, graph: nx.DiGraph, data_manager: DataManager, expand_all: bool = False):
        self.graph = graph
        self.data_manager = data_manager
        if expand_all:
            self.expanded_nodes: Set[int] = {n for n in graph.nodes() if graph.out_degree(n) > 0}
        else:
            self.expanded_nodes = set()
        
    def to_graph_node_id(self, cy_id: str) -> Union[int, str]:
        """Convert Cytoscape element id back to original graph node id."""
        try:
            return int(cy_id)
        except (ValueError, TypeError):
            return cy_id

    def get_descendants(self, node: int) -> Set[int]:
        """Get all descendants of a node."""
        return set(nx.descendants(self.graph, node))

    def get_visible_elements(self) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """Get elements that should be visible based on expanded state."""
        visible_nodes = set()
        visible_edges = set()
        
        # Always show root nodes
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        visible_nodes.update(roots)
        
        # Show nodes based on expansion state
        for node in self.expanded_nodes:
            children = list(self.graph.successors(node))
            visible_nodes.update(children)
        
        # Add edges between visible nodes
        for u, v in self.graph.edges():
            if u in visible_nodes and v in visible_nodes:
                visible_edges.add((u, v))
        
        return visible_nodes, visible_edges

    def build_elements(self) -> List[Dict]:
        """Build Cytoscape elements based on current state."""
        visible_nodes, visible_edges = self.get_visible_elements()
        elements = []
        
        # Add nodes
        for node in visible_nodes:
            node_data = {"id": str(node), "label": self.graph.nodes[node].get("node_name", str(node))}
            classes = []
            
            if self.graph.in_degree(node) == 0:
                classes.append("root")
            
            if node in self.expanded_nodes:
                classes.append("expanded")
            elif list(self.graph.successors(node)):
                classes.append("collapsed")
            
            element = {"data": node_data}
            if classes:
                element["classes"] = " ".join(classes)
            
            elements.append(element)
        
        # Add edges
        for u, v in visible_edges:
            elements.append({"data": {"source": str(u), "target": str(v)}})

        return elements

    def get_node_info(self, node_id: int) -> Dict:
        """Get information for a node to display in the side panel."""
        node_data = self.graph.nodes[node_id]
        is_leaf = len(list(self.graph.successors(node_id))) == 0
        
        if is_leaf:
            return self._get_leaf_node_info(node_data)
        else:
            return self._get_internal_node_info(node_id, node_data)
    
    def _get_leaf_node_info(self, node_data: Dict) -> Dict:
        """Get information for leaf nodes."""
        labels = node_data.get("all_associated_labels_in_subDAG", [])
        feature_count = len(labels)
        node_name = node_data.get('node_name', 'Unknown')
        associated_original_sae_ids = node_data.get("associated_original_sae_ids")
        associated_sae_subset_id = node_data.get("associated_sae_subset_ids")
        image_data_uris = [
            self.data_manager.get_image_data_uri(sid)
            for sid in node_data.get("all_associated_original_sae_ids_in_subDAG", [])
        ]
        image_data_uris = [uri for uri in image_data_uris if uri is not None]

        return {
            'type': 'leaf',
            'node_name': node_name,
            "all_associated_labels_in_subDAG": labels,
            'feature_count': feature_count,
            'image_data_uris': image_data_uris,
            'associated_sae_subset_id': associated_sae_subset_id[0] if associated_sae_subset_id else None,
            'associated_original_sae_id': associated_original_sae_ids[0] if associated_original_sae_ids else None
        }
    
    def _get_internal_node_info(self, node_id: int, node_data: Dict) -> Dict:
        """Get information for internal nodes."""
        node_dict = self.graph.nodes()[node_id]
        associated_original_sae_ids = node_dict.get("associated_original_sae_ids")
        associated_sae_subset_id = node_dict.get("associated_sae_subset_ids")
        print(f"Associated SAE IDs for node {node_id}: {associated_original_sae_ids}")
        all_associated_labels_in_subDAG = node_dict.get("all_associated_labels_in_subDAG", [])
        all_associated_sae_ids_in_subDAG = node_dict.get("all_associated_original_sae_ids_in_subDAG", [])
        assert len(all_associated_labels_in_subDAG) == len(all_associated_sae_ids_in_subDAG)
        
        node_name = node_data.get('node_name', 'Unknown')

        # Get direct feature image (max one)
        direct_image_data_uri = None
        if associated_original_sae_ids:
            direct_image_data_uri = self.data_manager.get_image_data_uri(associated_original_sae_ids[0])

        # Get representative labels
        necessary_num_elements = max(MAX_REPRESENTATIVE_LABELS, MAX_IMAGES_DISPLAY)
        random.seed(0)
        random_positions = random.sample(range(len(all_associated_labels_in_subDAG)), min(len(all_associated_labels_in_subDAG), necessary_num_elements))
        representative_labels = [all_associated_labels_in_subDAG[i] for i in random_positions[:MAX_REPRESENTATIVE_LABELS]]

        # Get representative images
        _representative_sae_ids = [all_associated_sae_ids_in_subDAG[i] for i in random_positions[:MAX_IMAGES_DISPLAY]]
        image_data_uris = [
            self.data_manager.get_image_data_uri(sid)
            for sid in _representative_sae_ids
        ]
        image_data_uris = [uri for uri in image_data_uris if uri is not None]

        return {
            'type': 'internal',
            'node_name': node_name,
            'direct_image_data_uri': direct_image_data_uri,
            'representative_labels': representative_labels,
            'total_features': len(all_associated_labels_in_subDAG),
            'image_data_uris': image_data_uris,
            'associated_sae_subset_id': associated_sae_subset_id[0] if associated_sae_subset_id else None,
            'associated_original_sae_id': associated_original_sae_ids[0] if associated_original_sae_ids else None
        }

    def create_side_panel(self, selected_node: Optional[int] = None) -> html.Div:
        """Create the side panel content based on selected node."""
        base_style = {
            'width': '300px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'border': '1px solid #dee2e6',
            'borderRadius': '5px',
            'height': '90vh',
            'overflowY': 'auto'
        }
        
        if selected_node is None:
            return html.Div([
                html.H4("Node Information"),
                html.P("Select a node to view details", style={'color': '#666'})
            ], style=base_style)
        
        node_info = self.get_node_info(selected_node)
        content = [html.H4(f"Node: {selected_node}"), html.Hr()]
        
        if node_info['type'] == 'leaf':
            content.extend(self._create_leaf_panel_content(node_info))
        else:
            content.extend(self._create_internal_panel_content(node_info))
        
        return html.Div(content, style=base_style)
    
    def _create_leaf_panel_content(self, node_info: Dict) -> List:
        """Create content for leaf node panel."""
        content = [
            html.H5("Leaf Node"),
            html.H6(f"Name: {node_info['node_name']}", style={'color': '#2a9d8f'}),
        ]

        # Add associated IDs if available
        if node_info.get('associated_sae_subset_ids') is not None:
            content.append(html.P(f"Associated SAE Subset ID: {node_info['associated_sae_subset_ids']}", style={'fontSize': '12px'}))
        if node_info.get('associated_original_sae_id') is not None:
            content.append(html.P(f"Associated Original SAE ID: {node_info['associated_original_sae_id']}", style={'fontSize': '12px'}))

        content.extend([
            html.P(f"Features: {node_info['feature_count']}"),
            html.Hr(),
            html.H6("SAE Feature Labels:"),
            html.Div([
                html.P(label, style={'margin': '2px 0', 'fontSize': '12px'})
                for label in node_info["all_associated_labels_in_subDAG"][:MAX_LABELS_DISPLAY]
            ], style={'maxHeight': '200px', 'overflowY': 'auto', 'border': '1px solid #ccc', 'padding': '10px'})
        ])
        
        # Add images if available
        if node_info['image_data_uris']:
            content.extend(self._create_image_section(node_info['image_data_uris'][:SINGLE_IMAGE_DISPLAY]))

        return content

    def _create_internal_panel_content(self, node_info: Dict) -> List:
        """Create content for internal node panel."""
        content = [
            html.H5("Internal Node"),
            html.H6(f"Name: {node_info['node_name']}", style={'color': '#2a9d8f'}),
            html.P(f"Total Features: {node_info['total_features']}")
        ]
        
        # Add associated IDs if available
        if node_info.get('associated_sae_subset_id') is not None:
            content.append(html.P(f"Associated SAE Subset ID: {node_info['associated_sae_subset_id']}", style={'fontSize': '12px'}))
        if node_info.get('associated_original_sae_id') is not None:
            content.append(html.P(f"Associated Original SAE ID: {node_info['associated_original_sae_id']}", style={'fontSize': '12px'}))
        
        # Add direct feature image if available
        if node_info.get('direct_image_data_uri'):
            content.extend(self._create_image_section([node_info['direct_image_data_uri']], description="Direct Feature Image"))

        content.extend([
            html.Hr(),
            html.H6("Representative Labels:"),
            html.Div([
                html.P(label, style={'margin': '2px 0', 'fontSize': '12px', 'color': '#666'})
                for label in node_info['representative_labels']
            ], style={'maxHeight': '200px', 'overflowY': 'auto', 'border': '1px solid #ccc', 'padding': '10px'})
        ])

        # Add representative images if available
        if node_info['image_data_uris']:
            content.extend(self._create_image_section(node_info['image_data_uris']))

        return content

    def _create_image_section(self, image_data_uris: List[str], description: str = "Top Activating Images") -> List:
        """Create image display section."""
        valid_images = [
            html.Img(
                src=uri,
                style={'width': f'{IMAGE_DISPLAY_WIDTH_PX}px', 'height': IMAGE_DISPLAY_HEIGHT, 'margin': '2px'}
            )
            for uri in image_data_uris if uri
        ]

        if valid_images:
            return [
                html.Hr(),
                html.H6(description),
                html.Div(valid_images, style={'display': 'flex', 'flexWrap': 'wrap'})
            ]
        return []

    def get_cytoscape_stylesheet(self) -> List[Dict]:
        """Get the stylesheet for Cytoscape."""
        return [
            {"selector": "node", 
             "style": {
                 "label": "data(label)", 
                 "width": 20, 
                 "height": 20, 
                 "background-color": "#69b",
                 "text-valign": "center",
                 "text-halign": "center",
                 "font-size": "8px"
             }},
            {"selector": "edge", 
             "style": {
                 "curve-style": "bezier", 
                 "target-arrow-shape": "triangle",
                 "arrow-color": "#666"
             }},
            {"selector": ".root", 
             "style": {"background-color": "#e76f51"}},
            {"selector": ".expanded", 
             "style": {
                 "background-color": "#2a9d8f",
                 "border-width": 2,
                 "border-color": "#264653"
             }},
            {"selector": ".collapsed", 
             "style": {
                 "background-color": "#f4a261",
                 "shape": "diamond"
             }},
            {"selector": "node:selected", 
             "style": {"border-width": 3, "border-color": "#000"}}
        ]

    def create_layout(self) -> html.Div:
        """Create the main Dash layout."""
        return html.Div([
            dcc.Store(id="selected-node", data=None),
            html.Div([
                cyto.Cytoscape(
                    id="cy",
                    elements=self.build_elements(),
                    style={"width": "70%", "height": "90vh"},
                    layout={"name": "dagre", "rankDir": "TB"},
                    stylesheet=self.get_cytoscape_stylesheet(),
                ),
                html.Div(id="side-panel", children=self.create_side_panel())
            ], style={'display': 'flex'}),
            html.Div(id="tap-output", style={"margin": "10px"})
        ])

    def handle_node_interaction(self, tap_data: Optional[Dict], selected_node: Optional[str]) -> Tuple:
        """Handle node selection and expansion/collapse logic."""
        if tap_data is None:
            return (
                self.build_elements(), 
                "Click to select, click selected node again to expand/collapse", 
                self.create_side_panel(), 
                None
            )

        node_id = self.to_graph_node_id(tap_data["id"])
        
        if selected_node == tap_data["id"]:
            return self._handle_node_toggle(node_id, tap_data)
        else:
            return self._handle_node_selection(node_id, tap_data)
    
    def _handle_node_toggle(self, node_id: int, tap_data: Dict) -> Tuple:
        """Handle toggling expansion state of selected node."""
        children = list(self.graph.successors(node_id))

        if not children:
            message = f"Leaf node '{node_id}' - cannot expand"
            side_panel = self.create_side_panel(node_id)
            return no_update, message, side_panel, tap_data["id"]

        if node_id in self.expanded_nodes:
            self.expanded_nodes.remove(node_id)
            descendants = self.get_descendants(node_id)
            self.expanded_nodes -= descendants
            action = "collapsed"
        else:
            self.expanded_nodes.add(node_id)
            action = "expanded"

        message = f"Node '{node_id}' {action}. Children: {len(children)}"
        elements = self.build_elements()
        side_panel = self.create_side_panel(node_id)
        return elements, message, side_panel, tap_data["id"]

    def _handle_node_selection(self, node_id: int, tap_data: Dict) -> Tuple:
        """Handle selection of new node."""
        message = f"Selected node '{node_id}'"
        side_panel = self.create_side_panel(node_id)
        return no_update, message, side_panel, tap_data["id"]


def create_app(graph_path, expand_all: bool = False) -> Dash:
    """Create and configure the Dash application."""
    # Initialize data manager and load data
    data_manager = DataManager(DEFAULT_CONFIG.graph_eval_dataset)
    data_manager.save_top_activating_image_paths()

    # Load graph
    graph = data_manager.load_graph(graph_path)
    graph.remove_nodes_from(list(nx.isolates(graph)))

    # Load extra layouts
    cyto.load_extra_layouts()

    # Create UI instance
    graph_ui = GraphUI(graph, data_manager, expand_all=expand_all)
    
    # Create Dash app
    app = Dash(__name__)
    app.layout = graph_ui.create_layout()
    
    # Define callback
    @app.callback(
        [
            Output("cy", "elements"), 
            Output("tap-output", "children"), 
            Output("side-panel", "children"), 
            Output("selected-node", "data")
        ],
        [Input("cy", "tapNodeData")],
        [State("selected-node", "data")]
    )
    def update_elements(tap_data, selected_node):
        return graph_ui.handle_node_interaction(tap_data, selected_node)
    
    return app


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Graph UI for visualizing hierarchical clustering results of SAE features.")
    # parser.add_argument("--graph_path", "-g", type=str, required=True,
    #                     help="Path to the graph file (PKL format)")
    parser.add_argument("--graph_name", "-g", type=str, required=True,
                    help="Name of the graph file wo extension (NOT full path, just the filename)")
    parser.add_argument("--port", "-p", type=int, default=6006,
                        help="Port to run the Dash app on")
    parser.add_argument("--expand-all", action="store_true",
                        help="Expand the entire hierarchy at startup instead of showing only roots")
    return parser.parse_args()

def main() -> None:
    """Main entry point for the application."""
    args = parse_args()
    graph_path = os.path.join(HIERARCHY_GRAPHS_PATH, f"{args.graph_name}.pkl")
    app = create_app(graph_path, expand_all=args.expand_all)
    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
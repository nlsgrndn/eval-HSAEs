from my_config import DEFAULT_CONFIG #CLIP_EMBEDDINGS_PATH, MODEL_NAME, SAE_FILTERING_STRATEGY, _INFERENCE_DATASET_STR
import os



MAIN_OUTPUT_FOLDER = "output"
INTERPRET_SUB_FOLDER = "interpret"
LABELING_OUTPUT_SUB_FOLDER = "labeling"
PURE_INTERPRETABILITY_METRICS_SUB_FOLDER = "interp_metrics"
SAE_IMAGES_SUB_FOLDER = "sae_images"
MONOSEMANTICITY_SUB_FOLDER = "monosemanticity"
CLARITY_SUB_FOLDER = "clarity"
LCA_IMAGENET_HIERARCHY_LEVEL_SUB_FOLDER = "lca_imagenet_hierarchy_level"
VECTOR_SEARCH_RESULTS_SUB_FOLDER = "vector_search_results"
MODEL_SPECIFIC_SUB_FOLDER = "model_specific_vocabs"
STRUCTURE_EXTRACTION_SUB_FOLDER = "structure_extraction"
DATASET_DATASTRUCTURES = "dataset_datastructures"
FAISS_INDICES = "faiss_indices"
GENERAL_INDICES = "general_indices"
EMBEDDINGS_FOLDER = "data"
SAE_ACTIVATIONS_FOLDER = "sae_activations"
INTERPRETABILITY_CSV_FILENAME = "summary_table_all_only_words_no_imagenet.csv"


class PathBuilder():
    
    def __init__(self, base_path = ".", config = DEFAULT_CONFIG):
        self.base_path = base_path
        self.config = config

    
    def get_embeddings_path(self):
        p = os.path.join(self.base_path, EMBEDDINGS_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_sae_activations_path(self):
        p = os.path.join(self.base_path, SAE_ACTIVATIONS_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_main_output_path(self):
        p = os.path.join(self.base_path, MAIN_OUTPUT_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_model_output_path(self):
        p = os.path.join(self.get_main_output_path(), self.config.sae_model.name)
        os.makedirs(p, exist_ok=True)
        return p

    def get_standard_sae_metrics_path(self):
        p = os.path.join(self.get_model_output_path(), "standard_evals")
        os.makedirs(p, exist_ok=True)
        return p

    def get_inference_dataset_path(self):
        p = os.path.join(self.get_model_output_path(), self.config.graph_eval_dataset.name)
        os.makedirs(p, exist_ok=True)
        return p

    def get_interpret_output_path(self):
        p = os.path.join(self.get_inference_dataset_path(), INTERPRET_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_labeling_output_path(self):
        p = os.path.join(self.get_interpret_output_path(), LABELING_OUTPUT_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_interpretability_data_path(self):
        file_path = os.path.join(self.get_labeling_output_path(), INTERPRETABILITY_CSV_FILENAME)
        return file_path

    def get_pure_interpretability_metrics_path(self):
        p = os.path.join(self.get_interpret_output_path(), PURE_INTERPRETABILITY_METRICS_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_monosemanticity_path(self):
        p = os.path.join(self.get_pure_interpretability_metrics_path(), MONOSEMANTICITY_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_clarity_path(self):
        p = os.path.join(self.get_pure_interpretability_metrics_path(), CLARITY_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_lca_imagenet_hierarchy_level_path(self):
        p = os.path.join(self.get_pure_interpretability_metrics_path(), LCA_IMAGENET_HIERARCHY_LEVEL_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_sae_images_path(self):
        p = os.path.join(self.get_interpret_output_path(), SAE_IMAGES_SUB_FOLDER, f"{self.config.graph_eval_dataset.name}_{self.config.graph_eval_dataset.split}")
        os.makedirs(p, exist_ok=True)
        return p

    def get_vector_search_results_path(self):
        p = os.path.join(self.get_interpret_output_path(), VECTOR_SEARCH_RESULTS_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_shared_vocab_path(self):
        p = os.path.join(self.base_path, "vocab")
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_model_specific_vocab_path(self):
        p = os.path.join(self.get_interpret_output_path(), MODEL_SPECIFIC_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_structure_extraction_path(self):
        p = os.path.join(self.get_inference_dataset_path(), STRUCTURE_EXTRACTION_SUB_FOLDER)
        os.makedirs(p, exist_ok=True)
        return p

    def get_structure_extraction_path_with_filtering_strategy(self):
        p = os.path.join(self.get_structure_extraction_path(), self.config.sae_filtering_strategy)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_geometry_analysis_path(self):
        p = os.path.join(self.get_structure_extraction_path(), "geometry_analysis")
        os.makedirs(p, exist_ok=True)
        return p

    def get_hierarchical_graphs_path(self):
        p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "hierarchical_graphs")
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_hierarchical_graph_eval_path(self):
        p = os.path.join(self.get_hierarchical_graphs_path(), self.config.graph_name)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_hierarchical_graph_aggregated_metrics_file_path(self):
        p = os.path.join(self.get_hierarchical_graph_eval_path(), "aggregated_metrics.pkl")
        return p
    
    def get_hierarchical_graph_eval_metadata_file_path(self):
        p = os.path.join(self.get_hierarchical_graph_eval_path(), "metadata_metrics.pkl")
        return p
    
    def get_conditional_activations_path(self):
        # p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "conditional_activations/sae_activations_metrics")
        p = os.path.join(self.get_structure_extraction_path(), "conditional_activations/sae_activations_metrics")
        os.makedirs(p, exist_ok=True)
        return p
    

    def get_cluster_labels_path(self):
        p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "cluster_labels")
        os.makedirs(p, exist_ok=True)
        return p

    def get_eval_per_approach_for_structure_extraction_path(self):
        p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "eval_per_approach")
        os.makedirs(p, exist_ok=True)
        return p

    def get_hierarchical_graph_visualizations_path(self):
        p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "vis_hierarchy_graphs")
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_vis_2d_embedding_space_path(self):
        p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "vis_2d_embedding_space")
        os.makedirs(p, exist_ok=True)
        return p

    def get_dataset_data_structures_path(self):
        p = os.path.join(self.base_path, DATASET_DATASTRUCTURES)
        os.makedirs(p, exist_ok=True)
        return p

    def get_faiss_indices_path(self):
        p = os.path.join(self.get_dataset_data_structures_path(), FAISS_INDICES)
        os.makedirs(p, exist_ok=True)
        return p

    def get_general_indices_path(self):
        p = os.path.join(self.get_dataset_data_structures_path(), GENERAL_INDICES)
        os.makedirs(p, exist_ok=True)
        return p

    def get_precomputed_avg_embeddings_path(self, top_k: int):
        folder = os.path.dirname(self.config.clip_embeddings_path)
        filename = os.path.basename(self.config.clip_embeddings_path).replace('.npy', f'_weighted_mean_top_{top_k}.npy')
        os.makedirs(os.path.join(folder, self.config.sae_model.name), exist_ok=True)
        return os.path.join(folder, self.config.sae_model.name, filename)
    
    def get_hierarchical_activations_path(self):
        p = os.path.join(self.get_structure_extraction_path_with_filtering_strategy(), "hierarchical_activations")
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_cbm_prediction_layer_path(self):
        p = os.path.join(self.get_inference_dataset_path(), "class_prediction_layer", self.config.sae_filtering_strategy)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_cbm_prediction_layer_model_weights_path(self):
        p = os.path.join(self.get_cbm_prediction_layer_path(), "model_weights")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
    
    def get_cbm_prediction_layer_eval_path(self):
        p = os.path.join(self.get_cbm_prediction_layer_path(), "evals")
        os.makedirs(p, exist_ok=True)
        return p



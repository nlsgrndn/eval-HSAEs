from my_config import DEFAULT_CONFIG
from valuable_notebook_code_snippets import (
    get_decoder_weights,
    get_encoder_weights,
)
from my_utils import load_dataset, load_embeddings_for_dataset, load_activations_for_dataset
from activations_preprocessing.utils_embeddings import load_avg_top_activation_images_embeddings
from label_assignment_strategies.load_sae_labelling_and_metrics_results import load_interpretability_data
from activations_preprocessing.utils_sae_activations import get_precomputed_top_k_path, load_precomputed_top_k
import numpy as np

class SAEDimensions:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def get_decoder_weights(self):
        decoder_weights = get_decoder_weights(self.config.sae_model.weights_path)
        return decoder_weights
    
    def get_encoder_weights(self):
        # assuming encoder weights are stored similarly to decoder weights
        encoder_weights = get_encoder_weights(self.config.sae_model.weights_path)
        return encoder_weights

    def get_avg_embeddings_on_graph_eval_dataset(self):
        avg_embeddings = load_avg_top_activation_images_embeddings(self.config)
        return avg_embeddings

    def get_activations_memmap_of_graph_creation_dataset(self):
        sae_activations = load_activations_for_dataset(self.config, subset="graph_creation_dataset")
        return sae_activations

    def get_activations_memmap_of_graph_evaluation_dataset(self):
        sae_activations = load_activations_for_dataset(self.config, subset="graph_eval_dataset")
        return sae_activations

    def get_interpretability_data(self):
        interpretability_data = load_interpretability_data(self.config)
        return interpretability_data
    
    def get_topk_activating_images_indices_and_values(self, TOPK=20):
        topk_path = get_precomputed_top_k_path(self.config.sae_latents_path)
        top_k_indices, top_k_values, _ = load_precomputed_top_k(topk_path)
        actual_topk = min(TOPK, top_k_indices.shape[1])
        return top_k_indices[:, :actual_topk], top_k_values[:, :actual_topk]

    def get_topk_embeddings_on_graph_eval_dataset(self, TOPK=20):
        topk_path = get_precomputed_top_k_path(self.config.sae_latents_path)
        top_k_indices, _, _ = load_precomputed_top_k(topk_path)
        embeddings_memmap = load_embeddings_for_dataset(self.config, subset="graph_eval_dataset")

        # estimate the necessary memory space for the result array
        # Memory estimate: num_latents * TOPK * embedding_dim * 4 bytes (float32)
        num_latents = top_k_indices.shape[0]
        embedding_dim = embeddings_memmap.shape[1]
        actual_topk = min(TOPK, top_k_indices.shape[1])
        estimated_memory_mb = (num_latents * TOPK * embedding_dim * 4) / (1024 * 1024)
        print(f"Estimated memory for topk embeddings: {estimated_memory_mb:.2f} MB")

        result = np.zeros((top_k_indices.shape[0], actual_topk, embeddings_memmap.shape[1]), dtype=np.float32)

        for sae_id in range((top_k_indices.shape[0])):
            indices = top_k_indices[sae_id, :actual_topk]
            embeddings = embeddings_memmap[indices, :]
            result[sae_id] = embeddings

        return result




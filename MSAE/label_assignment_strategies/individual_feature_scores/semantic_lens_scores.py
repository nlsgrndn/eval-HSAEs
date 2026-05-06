#copied from https://github.com/jim-berend/semanticlens
import numpy as np
import os
import torch
from sklearn.cluster import KMeans
from my_config import DEFAULT_CONFIG
from my_utils import load_embeddings_for_dataset
from label_assignment_strategies.individual_feature_scores.hierarchicality_scores_config import HierarchicalityScoresConfig
from activations_preprocessing.utils_sae_activations import get_precomputed_top_k_path, load_precomputed_top_k
from tqdm import tqdm
from path_hub import PathBuilder

_PB = PathBuilder()

@torch.inference_mode()
def clarity_score(V):
    """Clarity Score: How uniform are the concept examples => how clear is the representation?
    Higher is better in [-1/(n_samples-1), 1]
    Args:
        V (torch.Tesnor): (n_neurons) x n_samples x n_features
    Returns:
        torch.Tensor: (n_neurons) x 1
    """
    # V.shape = (n_neurons) x n_samples x n_features
    V_nrmed = torch.nn.functional.normalize(V, dim=-1)
    clarity = ((V_nrmed.mean(-2).pow(2).sum((-1))) - 1 / V.shape[-2]) / (V.shape[-2] - 1) * V.shape[-2]
    return clarity

def get_precomputed_top_k_and_embeddings_default():
    # load embeddings
    clip_embeddings = load_embeddings_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")

    print("Loading precomputed top-k activations...")
    # Load topk activating images per neuron
    top_k_indices, top_k_values, _ = load_precomputed_top_k(get_precomputed_top_k_path())
    return top_k_indices, top_k_values, clip_embeddings

if __name__ == "__main__":
    # Set parameters
    config = HierarchicalityScoresConfig()
    TOP_K = config.top_k_embeddings
    MINIMUM_NUMBER_OF_IMAGES_ABOVE_THRESHOLD = config.minimum_number_of_images_above_threshold
    ACTIVATION_THRESHOLD = config.activation_threshold
    
    # Load precomputed top-k activating images per neuron and clip embeddings
    top_k_indices, top_k_values, clip_embeddings = get_precomputed_top_k_and_embeddings_default()
    top_k_indices = top_k_indices[:, :TOP_K]
    top_k_values = top_k_values[:, :TOP_K]
    
    result_tensor = torch.zeros((top_k_indices.shape[0]), dtype=torch.float32)

    # collect embeddings and call clarity_score for each sae_id
    for sae_id in tqdm(range(top_k_indices.shape[0])):
        # get topk activations for sae_id
        indices_top_activations = top_k_indices[sae_id]
        values_top_activations = top_k_values[sae_id]

        # check if enough activations are above threshold
        num_images_above_threshold = (values_top_activations > ACTIVATION_THRESHOLD).sum()
        if num_images_above_threshold < MINIMUM_NUMBER_OF_IMAGES_ABOVE_THRESHOLD:
            result_tensor[sae_id] = float('nan')
            continue
        else:
            # get the embeddings for the sae_id
            embeddings_for_sae_id = clip_embeddings[indices_top_activations]
            embeddings_for_sae_id = torch.from_numpy(embeddings_for_sae_id)

            # compute clarity score
            clarity = clarity_score(embeddings_for_sae_id)
            result_tensor[sae_id] = clarity

    # save scores
    folder = _PB.get_clarity_path()
    os.makedirs(folder, exist_ok=True)
    torch.save(result_tensor, os.path.join(folder, "clarity_scores.pth"))
import numpy as np
import os
from tqdm import tqdm
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
from activations_preprocessing.utils_sae_activations import load_precomputed_top_k, get_precomputed_top_k_path
from my_utils import load_embeddings_for_dataset

def compute_weighted_mean_embeddings(clip_embeddings: np.ndarray, precomputed_path: str, 
                                   output_path: str, top_k: int):
    """
    Compute weighted mean embeddings for each SAE latent using top-k activating images.
    
    Args:
        clip_embeddings: Loaded CLIP embeddings array
        precomputed_path: Path to precomputed top-k indices and values
        output_path: Path to save weighted mean embeddings
        top_k: Number of top activating images to use
    """
    
    top_k_indices, top_k_values, max_activations = load_precomputed_top_k(precomputed_path)
    num_latents, _ = top_k_indices.shape
    
    num_samples, embedding_dim = clip_embeddings.shape
    
    # Initialize output array for weighted mean embeddings
    weighted_mean_embeddings = np.zeros((num_latents, embedding_dim), dtype=np.float32)
    
    # Compute weighted mean for each SAE latent
    for sae_id in tqdm(range(num_latents), desc="Computing weighted means"):
        # Get top-k indices and values for this latent
        indices = top_k_indices[sae_id][:top_k]
        values = top_k_values[sae_id][:top_k]
            
        # Retrieve relevant embeddings using the loaded indices
        relevant_embeddings = clip_embeddings[indices]  # Shape: (num_valid, embedding_dim)
        
        # Normalize weights to sum to 1
        weights = values / np.sum(values)  # Shape: (num_valid,)
        
        # Compute weighted mean of the retrieved embeddings (weigh by activation)
        weighted_mean = np.average(relevant_embeddings, axis=0, weights=weights)
        weighted_mean_embeddings[sae_id] = weighted_mean
    
    # Save weighted mean embeddings
    print(f"Saving weighted mean embeddings to {output_path}...")
    np.save(output_path, weighted_mean_embeddings)
    print(f"Saved weighted mean embeddings with shape {weighted_mean_embeddings.shape}")

def load_avg_top_activation_images_embeddings(config=DEFAULT_CONFIG):
    precomputed_avg_embeddings_path = PathBuilder(config=config).get_precomputed_avg_embeddings_path(DEFAULT_TOP_K)
    return np.load(precomputed_avg_embeddings_path)

DEFAULT_TOP_K = 20

def main():
    """
    Main function to compute weighted mean embeddings for both image and text embeddings.
    """
    top_activating_images_path = get_precomputed_top_k_path()

    output_path = PathBuilder().get_precomputed_avg_embeddings_path(DEFAULT_TOP_K)

    clip_embeddings = load_embeddings_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")

    compute_weighted_mean_embeddings(
        clip_embeddings=clip_embeddings,
        precomputed_path=top_activating_images_path,
        output_path=output_path,
        top_k=DEFAULT_TOP_K,
    )
    
    print(f"\nCompleted! Weighted mean embeddings saved to:")
    print(f"- embeddings: {output_path}")


if __name__ == "__main__":
    main()




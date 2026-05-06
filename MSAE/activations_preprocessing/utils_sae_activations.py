import numpy as np
from tqdm import tqdm
import os
import dotenv
dotenv.load_dotenv()

from my_config import DEFAULT_CONFIG
from my_utils import load_activations_for_dataset


def get_precomputed_top_k_path(path = None):
    if path is None:
        path = DEFAULT_CONFIG.sae_latents_path
    return path.replace('.npy', f'_top_{TOP_K_SAE_ACTIVATIONS}.npz')

def load_precomputed_top_k(precomputed_path: str):
    """Load precomputed top-k indices and values."""
    data = np.load(precomputed_path)
    return data['top_k_indices'], data['top_k_values'], data['max_activations']

def precompute_top_k_indices(sae_activations: np.ndarray, output_path: str, k: int = 100, batch_size: int = 10000):
    """
    Precompute top-k activating image indices for all SAE latents using row-wise batch processing.
    
    Args:
        sae_activations_path: Path to memory-mapped SAE activations
        output_path: Path to save precomputed indices
        k: Number of top indices to store per latent
        batch_size: Number of samples (rows) to process per batch
    """
    num_samples, num_latents = sae_activations.shape
    
    print(f"Processing {num_samples:,} samples across {num_latents:,} latents")
    print(f"Using row-wise batches of {batch_size:,} samples")
    
    # Preallocate results
    top_k_indices = np.zeros((num_latents, k), dtype=np.int32)
    top_k_values = np.zeros((num_latents, k), dtype=np.float32)
    max_activations = np.zeros(num_latents, dtype=np.float32)
    
    print(f"Computing top-{k} for {num_latents} latents...")
    
    # Process samples in row-wise batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing sample batches"):
        start_row = batch_idx * batch_size
        end_row = min(start_row + batch_size, num_samples)
        
        # Load batch of samples
        batch_data = sae_activations[start_row:end_row, :]  # Shape: (batch_size, num_latents)
        
        # Create global indices for this batch
        batch_indices = np.arange(start_row, end_row)
        
        # For each latent, update top-k if we find better values
        for sae_id in range(num_latents):
            latent_activations = batch_data[:, sae_id]
            
            # Combine current batch with existing top-k
            combined_values = np.concatenate([top_k_values[sae_id], latent_activations])
            combined_indices = np.concatenate([top_k_indices[sae_id], batch_indices])
            
            # Get top-k from combined values
            if len(combined_values) > k:
                top_k_mask = np.argpartition(combined_values, -k)[-k:]
                selected_values = combined_values[top_k_mask]
                selected_indices = combined_indices[top_k_mask]
                
                # Sort by value (descending)
                sorted_order = np.argsort(-selected_values)
                top_k_values[sae_id] = selected_values[sorted_order]
                top_k_indices[sae_id] = selected_indices[sorted_order]
            else:
                # Sort all values
                sorted_order = np.argsort(-combined_values)
                top_k_values[sae_id, :len(combined_values)] = combined_values[sorted_order]
                top_k_indices[sae_id, :len(combined_values)] = combined_indices[sorted_order]
            
            # Update max activation
            max_activations[sae_id] = top_k_values[sae_id, 0]
    
    # Save precomputed results
    if output_path is not None:
        np.savez_compressed(output_path,
                        top_k_indices=top_k_indices,
                        top_k_values=top_k_values,
                        max_activations=max_activations,
                        k=k)
    
    print(f"Saved precomputed top-{k} to {output_path}")
    return top_k_indices, top_k_values, max_activations

TOP_K_SAE_ACTIVATIONS = 100
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute top-k indices for SAE activations.")
    parser.add_argument('--batch_size', type=int, default=10000, help="Number of samples (rows) to process per batch")
    
    args = parser.parse_args()

    output_path = get_precomputed_top_k_path()
    sae_activations = load_activations_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")
    precompute_top_k_indices(sae_activations, output_path, TOP_K_SAE_ACTIVATIONS, args.batch_size)

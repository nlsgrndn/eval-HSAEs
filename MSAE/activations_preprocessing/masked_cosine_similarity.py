import numpy as np
import torch
from tqdm import tqdm
from utils_sae_feature_properties import SAEDimensions
from activations_preprocessing.act_behav_utils import precompute_binary_activations, preprocess_continuous_activations
from path_hub import PathBuilder
from configs.activation_preprocessing import get_acts_preprocess_cfg, ActivationsPreprocessingConfig
import os

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Compute masked cosine similarity for SAE features.")
    parser.add_argument('--max-gpu-samples', type=int, default=20000,
                        help="Maximum samples to load to GPU at once")
    parser.add_argument('--load-batch-size', type=int, default=10000,
                        help="Batch size for loading and binarizing from memmap")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use for computation (cuda or cpu)")
    return parser.parse_args()

def compute_cosine_row_gpu(acts_masked, feature_idx, device='cuda'):
    """
    Compute cosine similarities for one row (all j given masking feature i).
    
    Args:
        acts_masked: torch.Tensor of shape (n_active, n_features) on GPU
        feature_idx: int, the masking feature index (row index)
        device: str, device to use
    
    Returns:
        numpy array of shape (n_features,) with cosine similarities
    """
    n_active, n_features = acts_masked.shape
    
    if n_active == 0:
        return np.full(n_features, np.nan)
    
    # Get activations for masking feature i
    acts_i = acts_masked[:, feature_idx]  # (n_active,)
    
    # Check if feature i has zero norm
    if torch.norm(acts_i) < 1e-8:
        return np.full(n_features, np.nan)
    
    # Expand acts_i to match acts_masked shape for broadcasting
    acts_i_expanded = acts_i.unsqueeze(1).expand_as(acts_masked)  # (n_active, n_features)
    
    # Compute cosine similarities using PyTorch's built-in function
    cosine_sims = torch.nn.functional.cosine_similarity(
        acts_i_expanded, acts_masked, dim=0, eps=1e-8
    )  # (n_features,)
    
    return cosine_sims.cpu().numpy()


def compute_cosine_row_chunked_gpu(continuous_activations, mask_indices, feature_idx, 
                                   max_gpu_samples=500000, device='cuda'):
    """
    Compute cosine similarities for one row with sample-dimension chunking.
    
    Args:
        continuous_activations: numpy memmap of shape (n_samples, n_features)
        mask_indices: numpy array of active sample indices
        feature_idx: int, the masking feature index
        max_gpu_samples: int, maximum samples per GPU chunk
        device: str, device to use
    
    Returns:
        numpy array of shape (n_features,) with cosine similarities
    """
    n_active = len(mask_indices)
    n_features = continuous_activations.shape[1]
    
    if n_active == 0:
        return np.full(n_features, np.nan)
    
    # Calculate number of chunks needed
    n_chunks = (n_active + max_gpu_samples - 1) // max_gpu_samples
    
    # Accumulators for sufficient statistics
    dot_products = torch.zeros(n_features, dtype=torch.float32, device=device)
    sum_sq_i = torch.tensor(0.0, dtype=torch.float32, device=device)
    sum_sq_j = torch.zeros(n_features, dtype=torch.float32, device=device)
    
    # Process in chunks
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * max_gpu_samples
        end_idx = min((chunk_idx + 1) * max_gpu_samples, n_active)
        
        chunk_mask = mask_indices[start_idx:end_idx]
        
        # Load chunk to GPU
        acts_chunk = torch.from_numpy(
            continuous_activations[chunk_mask, :].astype(np.float32)
        ).to(device)
        
        acts_i_chunk = acts_chunk[:, feature_idx]  # (chunk_size,)
        
        # Accumulate squared norms
        sum_sq_i += (acts_i_chunk ** 2).sum()
        sum_sq_j += (acts_chunk ** 2).sum(dim=0)
        
        # Accumulate dot products
        dot_products += acts_i_chunk @ acts_chunk  # (n_features,)
        
        del acts_chunk, acts_i_chunk
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Compute final cosine similarities
    norm_i = torch.sqrt(sum_sq_i)
    norms_j = torch.sqrt(sum_sq_j)
    
    # Handle zero norms
    if norm_i < 1e-8:
        return np.full(n_features, np.nan)
    
    norms_j = torch.clamp(norms_j, min=1e-8)
    cosine_sims = dot_products / (norm_i * norms_j)
    
    return cosine_sims.cpu().numpy()

def compute_masked_cosine_similarity(config: ActivationsPreprocessingConfig, args):
    """
    Compute masked cosine similarity matrix for all SAE features.
    
    Args:
        config: ActivationsPreprocessingConfig for binarization strategy
        args: Command line arguments
    
    Returns:
        MCS matrix of shape (n_features, n_features)
    """
    # Load data
    sae_activations_memmap = SAEDimensions().get_activations_memmap_of_graph_creation_dataset()
    
    # Limit samples if specified
    n_samples = sae_activations_memmap.shape[0]
    if hasattr(config, 'max_num_samples') and config.max_num_samples is not None:
        n_samples = min(n_samples, config.max_num_samples)
    continuous_activations = sae_activations_memmap[:n_samples]
    continuous_activations = preprocess_continuous_activations(
        config,
        continuous_activations,
        args.load_batch_size
    )
    n_samples, n_features = continuous_activations.shape
    
    print(f"Computing MCS for {n_features} features across {n_samples} samples")
    print(f"Binarization: {config.binarization_strategy} with {config.binarization_kwargs}")
    print(f"Max GPU samples: {args.max_gpu_samples}")
    print(f"Device: {args.device}")
    
    # Initialize result matrix
    MCS = np.zeros((n_features, n_features), dtype=np.float32)
    base_rates = np.zeros(n_features, dtype=np.float32)

    binary_activations = precompute_binary_activations(
        config,
        sae_activations_memmap,
        args.load_batch_size
    )
    chunked_counter = 0
    # Process each masking feature
    for i in tqdm(range(n_features), desc="Computing MCS rows"):
        # Get mask for feature i
        mask = binary_activations[:, i]
        mask_indices = np.where(mask)[0]
        n_active = len(mask_indices)
        base_rates[i] = n_active / n_samples
        if n_active == 0:
            MCS[i, :] = np.nan
            continue
        
        # Adaptive strategy based on number of active samples
        if n_active < args.max_gpu_samples:
            # Fast path: Load all masked samples to GPU at once
            acts_masked = torch.from_numpy(
                continuous_activations[mask_indices, :]
            ).to(args.device)
            
            MCS[i, :] = compute_cosine_row_gpu(acts_masked, i, device=args.device)
            
            del acts_masked
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        else:
            chunked_counter += 1
            # Chunked path for dense features
            MCS[i, :] = compute_cosine_row_chunked_gpu(
                continuous_activations,
                mask_indices,
                i,
                max_gpu_samples=args.max_gpu_samples,
                device=args.device
            )
    print(f"Used chunked computation for {chunked_counter} features due to high density.")
    
    return MCS, base_rates


def save_mcs_matrix(mcs_matrix, base_rates, config: ActivationsPreprocessingConfig):
    """Save MCS matrix and base rates to disk."""
    output_dir = PathBuilder().get_conditional_activations_path()
    output_dir = os.path.join(output_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)
    
    mcs_path = os.path.join(output_dir, "masked_cosine_similarity.npy")
    base_rates_path = os.path.join(output_dir, "masked_cosine_similarity_base_rates.npy")
    
    np.save(mcs_path, mcs_matrix)
    np.save(base_rates_path, base_rates)
    
    print(f"Saved MCS matrix to: {mcs_path}")
    print(f"Saved base rates to: {base_rates_path}")


def load_mcs_matrix(config: ActivationsPreprocessingConfig, sae_config=None):
    """Load MCS matrix and base rates from disk."""
    input_dir = PathBuilder().get_conditional_activations_path() if sae_config is None else PathBuilder(config = sae_config).get_conditional_activations_path()
    input_dir = os.path.join(input_dir, config.name)
    
    mcs_path = os.path.join(input_dir, "masked_cosine_similarity.npy")
    base_rates_path = os.path.join(input_dir, "masked_cosine_similarity_base_rates.npy")
    
    if not os.path.exists(mcs_path):
        raise FileNotFoundError(f"MCS matrix not found at: {mcs_path}")
    if not os.path.exists(base_rates_path):
        raise FileNotFoundError(f"Base rates not found at: {base_rates_path}")
    
    mcs_matrix = np.load(mcs_path)
    base_rates = np.load(base_rates_path)
    
    print(f"Loaded MCS matrix from: {mcs_path}")
    print(f"Loaded base rates from: {base_rates_path}")
    
    return mcs_matrix, base_rates

def main():
    args = parse_args()
    config = get_acts_preprocess_cfg()
    
    print("Starting masked cosine similarity computation...")
    mcs_matrix, base_rates = compute_masked_cosine_similarity(config, args)
    
    print(f"MCS matrix shape: {mcs_matrix.shape}")
    print(f"Non-NaN entries: {np.sum(~np.isnan(mcs_matrix))}")
    print(f"Base rates shape: {base_rates.shape}")
    
    save_mcs_matrix(mcs_matrix, base_rates, config)
    print("Done!")


if __name__ == "__main__":
    main()

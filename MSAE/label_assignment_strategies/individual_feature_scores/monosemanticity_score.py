# code copied from https://github.com/ExplainableML/sae-for-vlm
import torch
import os.path
import argparse
#from datasets.activations import ActivationsDataset
import os
import numpy as np

from torch.utils.data import DataLoader, Subset, Dataset
import tqdm
import torch.nn.functional as F
import bisect
from my_config import DEFAULT_CONFIG
from my_utils import load_embeddings_for_dataset, load_activations_for_dataset
from path_hub import PathBuilder

def get_args_parser():
    parser = argparse.ArgumentParser("Measure monosemanticity via weighted pairwise cosine similarity", add_help=False)
    #parser.add_argument("--embeddings_path")
    parser.add_argument("--activations_dir", default="monosemanticity")
    parser.add_argument("--output_subdir", default="output")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--approximate", "-a", action='store_true', help="Use approximate method for faster computation")
    parser.add_argument("--max_num_data_points","-max_n" ,type=int, default=None)
    parser.add_argument("--batch_size_i","-bi",type=int, default=256, help="Chunk size for i values")
    parser.add_argument("--batch_size_j","-bj",type=int, default=1024, help="Chunk size for j values") #2**17 = 131072
    return parser

# Compute min/max efficiently in chunks to avoid loading full dataset
def compute_min_max_chunked(memmap_array, chunk_size=10000):
    num_samples, num_features = memmap_array.shape
    
    # Initialize with first chunk
    first_chunk = memmap_array[:min(chunk_size, num_samples)]
    global_min = np.min(first_chunk, axis=0)
    global_max = np.max(first_chunk, axis=0)
    
    # Process remaining chunks
    for start_idx in range(chunk_size, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk = memmap_array[start_idx:end_idx]
        
        chunk_min = np.min(chunk, axis=0)
        chunk_max = np.max(chunk, axis=0)
        
        global_min = np.minimum(global_min, chunk_min)
        global_max = np.maximum(global_max, chunk_max)
    
    return global_min, global_max

def main(args):
    max_num_data_points = args.max_num_data_points
    # Load embeddings
    embeddings_numpy_memmap = load_embeddings_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")
    embeddings_numpy_memmap = embeddings_numpy_memmap[:max_num_data_points] if max_num_data_points is not None else embeddings_numpy_memmap
    print(f"Loaded embeddings found at {DEFAULT_CONFIG.clip_embeddings_path}")
    print(f"Embeddings shape: {embeddings_numpy_memmap.shape}")

    # Load activations
    activations_numpy_memmap = load_activations_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")
    activations_numpy_memmap = activations_numpy_memmap[:max_num_data_points] if max_num_data_points is not None else activations_numpy_memmap
    print(f"Loaded activations found at {args.activations_dir}")
    print(f"Activations shape: {activations_numpy_memmap.shape}")

    # embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    num_images, embed_dim = embeddings_numpy_memmap.shape
    num_neurons = activations_numpy_memmap.shape[1]

    # Scale to 0-1 per neuron
    print("Computing min/max values efficiently...")
    min_values, max_values = compute_min_max_chunked(activations_numpy_memmap)
    value_ranges = max_values - min_values

    # Avoid division by zero
    value_ranges = np.where(value_ranges == 0, 1, value_ranges)

    min_values_on_device = torch.from_numpy(min_values).to(torch.device(args.device))
    value_ranges_on_device = torch.from_numpy(value_ranges).to(torch.device(args.device))

    # Don't modify the memmap in-place - normalize during processing instead
    def normalize_activations_torch(activations_tensor):
        """Normalize activation tensor on-the-fly"""
        return (activations_tensor - min_values_on_device) / value_ranges_on_device

    # Initialize accumulators
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=torch.device(args.device))
    weight_sum = torch.zeros(num_neurons, device=torch.device(args.device))
    
    # Chunked triangular processing parameters
    batch_size_i = args.batch_size_i  # Process chunks of i values
    batch_size_j = args.batch_size_j  # Batch size for j values
    #Note: only chunk_size affects how accurate the approximate computation is

    print(f"Using batch_size_i: {batch_size_i}, batch_size_j: {batch_size_j}")
    
    for i_start in tqdm.tqdm(range(0, num_images, batch_size_i), desc="Processing chunks"):
        i_end = min(i_start + batch_size_i, num_images)
        
        # Load chunk of embeddings and activations for i values
        embeddings_i_chunk = torch.from_numpy(
            embeddings_numpy_memmap[i_start:i_end].copy()
        ).to(torch.device(args.device))
        
        # activations_i_chunk = torch.from_numpy(
        #     normalize_activations(activations_numpy_memmap[i_start:i_end].copy())
        # ).to(torch.device(args.device))
        activations_i_chunk = normalize_activations_torch(torch.from_numpy(
            activations_numpy_memmap[i_start:i_end].copy()
        ).to(torch.device(args.device)))

        # Process pairs within the chunk (triangular part)
        actual_chunk_size = embeddings_i_chunk.shape[0]
        if not args.approximate and actual_chunk_size > 1:
            # Vectorized within-chunk processing
            cos_sim_matrix = torch.mm(
                F.normalize(embeddings_i_chunk, p=2, dim=1),
                F.normalize(embeddings_i_chunk, p=2, dim=1).t()
            )
            
            # Create upper triangular mask
            mask = torch.triu(torch.ones_like(cos_sim_matrix, dtype=torch.bool), diagonal=1)
            
            if mask.any():
                cos_sims = cos_sim_matrix[mask]
                
                # Compute weights for all pairs
                weights_matrix = activations_i_chunk.unsqueeze(1) * activations_i_chunk.unsqueeze(0)
                weights = weights_matrix[mask]
                
                weighted_similarities = weights * cos_sims.unsqueeze(1)
                
                weighted_cosine_similarity_sum += torch.sum(weighted_similarities, dim=0)
                weight_sum += torch.sum(weights, dim=0)
        
        # Process pairs between this chunk and all following data
        for j_start in range(i_end, num_images, batch_size_j):
            j_end = min(j_start + batch_size_j, num_images)
            
            embeddings_j = torch.from_numpy(
                embeddings_numpy_memmap[j_start:j_end].copy()
            ).to(torch.device(args.device))

            activations_j = normalize_activations_torch(torch.from_numpy(
                activations_numpy_memmap[j_start:j_end].copy()
            ).to(torch.device(args.device)))

            # With vectorized computation:
            cos_sim_matrix = torch.mm(
                F.normalize(embeddings_i_chunk, p=2, dim=1),
                F.normalize(embeddings_j, p=2, dim=1).t()
            )  # Shape: (actual_chunk_size, j_batch_size)

            # Compute all weights at once using broadcasting
            weights_matrix = activations_i_chunk.unsqueeze(1) * activations_j.unsqueeze(0)
            # Shape: (actual_chunk_size, j_batch_size, num_neurons)

            # Vectorized weighted similarities
            weighted_similarities = weights_matrix * cos_sim_matrix.unsqueeze(2)

            # Sum across all pairs
            weighted_cosine_similarity_sum += torch.sum(weighted_similarities.view(-1, num_neurons), dim=0)
            weight_sum += torch.sum(weights_matrix.view(-1, num_neurons), dim=0)


    weight_sum = weight_sum.cpu()  #
    weighted_cosine_similarity_sum = weighted_cosine_similarity_sum.cpu()

    monosemanticity = torch.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan)

    # Use PathBuilder to resolve monosemanticity base path if activations_dir is default
    out_dir = PathBuilder().get_monosemanticity_path()
    os.makedirs(out_dir, exist_ok=True)
    file_name = f"all_neurons_scores{f'_approximate_i{batch_size_i}_j{batch_size_j}' if args.approximate else ''}{f'MaxN{max_num_data_points}' if max_num_data_points is not None else ''}.pth"
    torch.save(monosemanticity, os.path.join(out_dir, file_name))

    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])

    print(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}")
    print(f"Dead neurons:", nan_count.item())
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()

    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, 10)
    bottom_10_values, bottom_10_indices = torch.topk(valid_monosemanticity, 10, largest=False)

    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]

    # Print results
    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    # Save to file
    file_name = f"metric_stats_new{f'_approximate_i{batch_size_i}_j{batch_size_j}' if args.approximate else ''}.txt"
    output_path = os.path.join(out_dir, file_name)
    with open(output_path, "w") as file:
        file.write(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}\n")
        file.write(f"Dead neurons: {nan_count.item()}\n")
        file.write(f"Total neurons: {num_neurons}\n\n")

        file.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_10_indices, top_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")

        file.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bottom_10_indices, bottom_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
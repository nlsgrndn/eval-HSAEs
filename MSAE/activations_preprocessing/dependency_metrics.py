import numpy as np
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import torch
import os
from configs.activation_preprocessing import get_acts_preprocess_cfg
from path_hub import PathBuilder


class SaveAndLoad:
    
    @staticmethod
    def save_data_as_npy_arrays(data_dict, cond_acts_metric_cfg=None, path_builder=None):
        if path_builder is None:
            path_builder = PathBuilder()
        output_dir = path_builder.get_conditional_activations_path()
        if cond_acts_metric_cfg is None:
            cond_acts_metric_cfg = get_acts_preprocess_cfg()
        output_dir = os.path.join(output_dir, cond_acts_metric_cfg.name)
        os.makedirs(output_dir, exist_ok=True)
        print("Saving dependency metrics for SAE activations...")
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for key, matrix in data_dict.items():
            np.save(os.path.join(output_dir, f"{key}.npy"), matrix)
    
    @staticmethod
    def load_data_from_npy_arrays(cond_acts_metric_cfg=None, path_builder=None):
        if path_builder is None:
            path_builder = PathBuilder()
        output_dir = path_builder.get_conditional_activations_path()
        if cond_acts_metric_cfg is None:
            cond_acts_metric_cfg = get_acts_preprocess_cfg()
        output_dir = os.path.join(output_dir, cond_acts_metric_cfg.name)
        data_dict = {}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.npy'):
                key = file_name[:-4]  # remove .npy extension
                matrix = np.load(os.path.join(output_dir, file_name))
                data_dict[key] = matrix
        return data_dict

def compute_dependency_metrics(binary_activations, continuous_activations, compute_r2=False, compute_correlation=False, compute_mutual_info=False, device='cuda', batch_size=50000):
    """
    Compute dependency metrics between all pairs of features based on binary activations.
    Uses a batched approach to handle large datasets and works on both CPU and GPU.
    
    Args:
        binary_activations: numpy array or torch tensor of shape (n_samples, n_features) with binary values
        continuous_activations: numpy array or torch tensor of shape (n_samples, n_features) with continuous values
        compute_r2: bool, whether to compute R² metric
        compute_correlation: bool, whether to compute correlation metric
        compute_mutual_info: bool, whether to compute mutual information
        device: str, device to use ('cuda' or 'cpu')
        batch_size: int, number of samples to process at once
    
    Returns:
        dict with matrices where [i,j] represents feature i (row) and feature j (column):
            - base_rates: Base activation rate for each feature (1D array)
            - P_j_given_i: P(j=1 | i=1)
            - conditional_activation_ratio: P(j|i) / P(j|~i)
            - directionality: P(i|j) - P(j|i)
            - lift: P(j|i) / P(j)
            - mutual_information: Mutual information between feature i and feature j (if compute_mutual_info=True)
            - r2: R² predicting feature j from feature i (if compute_r2=True)
            - correlation: Pearson correlation coefficient (if compute_correlation=True)
    """
    # Handle input type
    if isinstance(binary_activations, torch.Tensor):
        binary_activations = binary_activations.cpu().numpy()
    
    # Handle device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
        
    n_samples, n_features = binary_activations.shape
    
    # Initialize accumulators on device
    i_and_j_both = torch.zeros((n_features, n_features), dtype=torch.float32, device=device)
    total_activations = torch.zeros(n_features, dtype=torch.float32, device=device)
    i_and_j_weighted_by_i = torch.zeros((n_features, n_features), dtype=torch.float32, device=device)
    total_continuous_activations = torch.zeros(n_features, dtype=torch.float32, device=device)
    
    if compute_correlation or compute_r2:
        sum_x = torch.zeros(n_features, dtype=torch.float32, device=device)
        sum_x_squared = torch.zeros(n_features, dtype=torch.float32, device=device)
        sum_xy = torch.zeros((n_features, n_features), dtype=torch.float32, device=device)
    
    # Process in batches
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"Processing {n_samples} samples in {n_batches} batches of size {batch_size} on {device}")
    
    for batch_idx in tqdm(range(n_batches), desc=f"Processing batches on {device}"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        # Load batch to device
        # binary_activations is numpy, so we convert to tensor
        batch_binary = torch.from_numpy(binary_activations[start_idx:end_idx]).to(device).bool()
        batch_binary_float = batch_binary.float()

        
        # Accumulate co-occurrence matrix: binary_float.T @ binary_float
        i_and_j_both += batch_binary_float.T @ batch_binary_float
        
        # Accumulate activation counts
        total_activations += batch_binary.sum(dim=0).float()

        batch_continuous_float = torch.from_numpy(continuous_activations[start_idx:end_idx]).to(device).float()

        # Weight co-occurrence by continuous activations of i: (batch_continuous_float * batch_binary_float).T @ batch_binary_float
        batch_continuous_when_active = batch_continuous_float * batch_binary_float # zero out continuous values when i is not active; might not do anything if continuous values are already zero when inactive
        i_and_j_weighted_by_i += batch_continuous_when_active.T @ batch_binary_float # weight co-occurrence by continuous activations of i

        # Accumulate total continuous activations but only for active cases of i
        total_continuous_activations += batch_continuous_when_active.sum(dim=0)
        
        # Accumulate statistics for correlation if needed
        if compute_correlation or compute_r2:
            sum_x += batch_binary_float.sum(dim=0)
            sum_x_squared += (batch_binary_float ** 2).sum(dim=0)
            sum_xy += batch_binary_float.T @ batch_binary_float
        
        # Free batch memory
        del batch_binary, batch_binary_float, batch_continuous_float, batch_continuous_when_active
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Compute base rates
    base_rates = total_activations / n_samples
    # Compute counts for feature i (rows)
    i_counts = total_activations.unsqueeze(1)  # (n_features, 1)
    i_counts = torch.clamp(i_counts, min=1)

    # Compute continuous activation totals for feature i (rows)
    total_continuous_activations = total_continuous_activations.unsqueeze(1)  # (n_features, 1)
    total_continuous_activations = torch.clamp(total_continuous_activations, min=1e-8)
    
    # Compute P(j | i)
    P_j_given_i = i_and_j_both / i_counts

    # Compute weighted P(j | i) using continuous activations of i
    P_j_given_i_weighted = i_and_j_weighted_by_i / total_continuous_activations

    # Compute P(j | ~i)
    i_inactive_counts = n_samples - total_activations.unsqueeze(1)
    i_inactive_counts = torch.clamp(i_inactive_counts, min=1)
    
    j_activations_total = total_activations.unsqueeze(0)  # (1, n_features)
    j_when_i_inactive = j_activations_total.T - i_and_j_both
    P_j_given_not_i_matrix = j_when_i_inactive / i_inactive_counts
    
    result_dict = {
        'base_rates': base_rates.cpu().numpy(),
        'P_j_given_i': P_j_given_i.cpu().numpy(),
        'P_j_given_i_weighted': P_j_given_i_weighted.cpu().numpy(),
    }
    
    # Compute correlation if needed
    if compute_correlation or compute_r2:
        # Pearson correlation from sufficient statistics
        mean_x = sum_x / n_samples
        std_x = torch.sqrt(sum_x_squared / n_samples - mean_x ** 2)
        std_x = torch.clamp(std_x, min=1e-8)
        
        # Covariance matrix
        cov_matrix = sum_xy / n_samples - mean_x.unsqueeze(1) @ mean_x.unsqueeze(0)
        
        # Correlation matrix
        correlation = cov_matrix / (std_x.unsqueeze(1) @ std_x.unsqueeze(0))
        correlation.fill_diagonal_(float('nan'))
        
        if compute_correlation:
            result_dict['correlation'] = correlation.cpu().numpy()
        
        if compute_r2:
            r2 = correlation ** 2
            result_dict['r2'] = r2.cpu().numpy()
    
    # Mutual information (computed on CPU with batching)
    if compute_mutual_info:
        print("Computing mutual information on CPU (batched to save memory)...")
        mutual_information = np.full((n_features, n_features), np.nan)
        
        # We need to compute MI for all pairs - this is still O(n_features^2)
        # But we can at least batch the data loading
        for i in tqdm(range(n_features), desc="Computing mutual information"):
            for j in range(i + 1, n_features):
                # Accumulate counts for MI computation
                mi_val = mutual_info_score(binary_activations[:, i], binary_activations[:, j])
                mutual_information[i, j] = mi_val
                mutual_information[j, i] = mi_val
        
        result_dict['mutual_information'] = mutual_information
    
    # Compute derived metrics
    conditional_activation_ratio = P_j_given_i / torch.clamp(P_j_given_not_i_matrix, min=1e-8)
    conditional_activation_ratio = torch.where(
        P_j_given_not_i_matrix > 1e-8,
        conditional_activation_ratio,
        torch.tensor(float('nan'), device=device)
    )
    
    directionality = P_j_given_i.T - P_j_given_i # P(i|j) - P(j|i)
    
    # Compute Lift: P(j|i) / P(j)
    lift = P_j_given_i / torch.clamp(base_rates.unsqueeze(0), min=1e-8)
    lift = torch.where(
        base_rates.unsqueeze(0) > 1e-8,
        lift,
        torch.tensor(float('nan'), device=device)
    )
    
    result_dict['conditional_activation_ratio'] = conditional_activation_ratio.cpu().numpy()
    result_dict['directionality'] = directionality.cpu().numpy()
    result_dict['lift'] = lift.cpu().numpy()
    
    return result_dict
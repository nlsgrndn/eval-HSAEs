from sklearn.metrics.pairwise import cosine_similarity
from configs.activation_preprocessing import BinarizationStrategy, ActivationsPreprocessingConfig
import numpy as np
from tqdm import tqdm
from utils_sae_feature_properties import SAEDimensions

class SAEActivationsPostProcessing:

    PRECOMPUTE_MEANS_MIN_SAMPLES = 100000

    def __init__(self, sae_activations_config):
        self.sae_activations_config = sae_activations_config
        self.precomputed_means = None

    def precompute_means(self, A):
        # check if mean centering is enabled
        if self.sae_activations_config.mean_center:
            # check if means are already computed
            if self.precomputed_means is None:
                #check whether activations have min length
                if A.shape[0] >= SAEActivationsPostProcessing.PRECOMPUTE_MEANS_MIN_SAMPLES:
                    self.precomputed_means = A.mean(axis=0, keepdims=True)
                    print("Set precomputed means for mean centering.")
                else:
                    raise ValueError(f"Cannot precompute means: activations have {A.shape[0]} samples, need at least {SAEActivationsPostProcessing.PRECOMPUTE_MEANS_MIN_SAMPLES}")
            else:
                print("Means already precomputed.")
        else:
            print("Mean centering not enabled; skipping precomputation of means.")
            
    def postprocess(self, A):
        # mean center
        if self.sae_activations_config.mean_center:
            A = self._mean_center_activations(A)
        else:
            A = np.maximum(A, 0) # hacky way to ensure that it is not read only
        # apply top-k thresholding
        if self.sae_activations_config.apply_top_k_preprocess:
            A = self._apply_topk(A, self.sae_activations_config.apply_top_k_preprocess)
        return A

    def binarize(self, A):
        B = SAEActivationsPostProcessing._binarize_activations(A, self.sae_activations_config.binarization_strategy, **self.sae_activations_config.binarization_kwargs)
        return B
    
    def _apply_topk(self, A, top_k):
        binarization_params = {"top_k": top_k}
        binary_activations = SAEActivationsPostProcessing._binarize_activations(A, BinarizationStrategy.TOP_K, **binarization_params)
        A[~binary_activations.astype(bool)] = 0.0
        return A

    def _mean_center_activations(self, A):
        if self.precomputed_means is not None:
            mean = self.precomputed_means
        else:
            raise ValueError("Precomputed means are not available. Call precompute_means() first.")
        A = A - mean
        A = np.maximum(A, 0)
        return A
    
    @staticmethod
    def _binarize_activations(A, strategy, **binarization_params):
        if strategy == BinarizationStrategy.ABSOLUTE_THRESHOLD:
            return (A > binarization_params["absolute_threshold"]).astype(bool)
        elif strategy == BinarizationStrategy.TOP_K:
            # for each datapoint, apply top-k thresholding. use vectorized implementation
            k = binarization_params["top_k"]
            k_clamped = min(k, A.shape[1])

            if k == A.shape[1]:
                return (A > 0).astype(bool)
            
            # increase k by one to do greater-than comparison
            k_clamped += 1
            
            # Get k-th largest value for each row
            kth_values = np.partition(A, -k_clamped, axis=1)[:, -k_clamped].reshape(-1, 1)
            # Only mark features active if they exceed the threshold AND are positive
            return ((A > kth_values) & (A > 0)).astype(bool)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def precompute_binary_activations(cond_act_metrics_config: ActivationsPreprocessingConfig, sae_activations_memmap, load_batch_size, order ='C'):
    total_samples = min(cond_act_metrics_config.max_num_samples, sae_activations_memmap.shape[0])
    n_features = sae_activations_memmap.shape[1]
    n_load_batches = (total_samples + load_batch_size - 1) // load_batch_size
    
    print(f"Processing {total_samples} samples in {n_load_batches} loading batches")
    print(f"Load batch size (CPU): {load_batch_size}")
    print(f"Binarization: {cond_act_metrics_config.binarization_strategy} with {cond_act_metrics_config.binarization_kwargs}")
    print(f"Pre-allocating binary array: {total_samples} × {n_features} = {total_samples * n_features / (1024**3):.2f} GB")
    binary_activations = np.empty((total_samples, n_features), dtype=bool, order=order)
    
    postprocessor = SAEActivationsPostProcessing(cond_act_metrics_config)
    postprocessor.precompute_means(sae_activations_memmap)
    for batch_idx in tqdm(range(n_load_batches), desc="Loading and binarizing batches"):
        start_idx = batch_idx * load_batch_size
        end_idx = min((batch_idx + 1) * load_batch_size, total_samples)
        
        batch = sae_activations_memmap[start_idx:end_idx,]

        batch = postprocessor.postprocess(batch)
        binary_activations[start_idx:end_idx] = postprocessor.binarize(batch)
        
        del batch
    
    print(f"Binary activations shape: {binary_activations.shape}")
    print(f"Binary activations memory: {binary_activations.nbytes / (1024**3):.2f} GB")
    return binary_activations

def preprocess_continuous_activations(cond_act_metrics_config: ActivationsPreprocessingConfig, sae_activations_memmap, load_batch_size):
    total_samples = min(cond_act_metrics_config.max_num_samples, sae_activations_memmap.shape[0])
    n_features = sae_activations_memmap.shape[1]
    n_load_batches = (total_samples + load_batch_size - 1) // load_batch_size
    
    print(f"Processing {total_samples} samples in {n_load_batches} loading batches")
    print(f"Load batch size (CPU): {load_batch_size}")
    print(f"Preprocessing: {cond_act_metrics_config.name}")
    print(f"Pre-allocating continuous array: {total_samples} × {n_features} = {total_samples * n_features / (1024**3):.2f} GB")
    processed_activations = np.empty((total_samples, n_features), dtype=np.float32)
    
    postprocessor = SAEActivationsPostProcessing(cond_act_metrics_config)
    postprocessor.precompute_means(sae_activations_memmap)
    for batch_idx in tqdm(range(n_load_batches), desc="Loading and preprocessing batches"):
        start_idx = batch_idx * load_batch_size
        end_idx = min((batch_idx + 1) * load_batch_size, total_samples)
        
        batch = sae_activations_memmap[start_idx:end_idx,]

        batch = postprocessor.postprocess(batch)
        processed_activations[start_idx:end_idx] = batch.astype(np.float32)
        
        del batch
    
    print(f"Processed activations shape: {processed_activations.shape}")
    print(f"Processed activations memory: {processed_activations.nbytes / (1024**3):.2f} GB")
    return processed_activations


def build_similarity_nonneg(Z):
    # cosine on z-scored columns == Pearson corr in [-1,1]
    S = cosine_similarity(Z.T)
    # rescale to [0,1]; preserves ordering, avoids negatives
    S = (S + 1.0) / 2.0
    np.fill_diagonal(S, 0.0)
    return S

def zscore_columns(A, eps=1e-8):
    # z-score each feature across images (columns)
    mu = A.mean(axis=0)
    sd = A.std(axis=0) + eps
    return (A - mu) / sd
from utils_sae_feature_properties import SAEDimensions
from activations_preprocessing.act_behav_utils import precompute_binary_activations, preprocess_continuous_activations
from path_hub import PathBuilder
from activations_preprocessing.dependency_metrics import compute_dependency_metrics, SaveAndLoad
from configs.activation_preprocessing import get_acts_preprocess_cfg, ActivationsPreprocessingConfig
import os
import psutil

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Compute and analyze conditional activations for SAE dimensions in clusters.")
    parser.add_argument('--load-batch-size', type=int, default=10000,
                        help="Batch size for loading and binarizing from memmap")
    parser.add_argument('--gpu-batch-size', type=int, default=50000,
                        help="Batch size for GPU computation")
    return parser.parse_args()

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.1f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.1f} GB")
    print(f"Used RAM: {mem.used / (1024**3):.1f} GB")
    print(f"Memory usage: {mem.percent}%")

def main(cond_act_metrics_config: ActivationsPreprocessingConfig, args):
    """
    Batch-wise processing version for large-scale datasets.
    """
    check_memory()
    
    sae_activations_memmap = SAEDimensions().get_activations_memmap_of_graph_creation_dataset()

    continuous_activations = preprocess_continuous_activations(
        cond_act_metrics_config,
        sae_activations_memmap,
        args.load_batch_size
    )
    
    binary_activations = precompute_binary_activations(
        cond_act_metrics_config,
        sae_activations_memmap,
        args.load_batch_size
    )
    
    output_base_dir = PathBuilder().get_conditional_activations_path()
    os.makedirs(output_base_dir, exist_ok=True)
    

    print(f"GPU batch size: {args.gpu_batch_size}")
    dependency_metrics = compute_dependency_metrics(
        binary_activations,
        continuous_activations,
        batch_size=args.gpu_batch_size
    )
    SaveAndLoad.save_data_as_npy_arrays(dependency_metrics, cond_acts_metric_cfg=cond_act_metrics_config)

if __name__ == "__main__":
    args = parse_args()
    
    cond_act_metrics_config = get_acts_preprocess_cfg()  # Example config name
    
    main(cond_act_metrics_config, args)
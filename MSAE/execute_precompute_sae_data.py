import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder

@dataclass
class PipelineConfig:
    batch_size: int
    sae_model_weights_path: str
    model_embeddings_path: str 
    output_dir: str
    subset_size: str


def run_precompute_pipeline(config: PipelineConfig):
    """
    Run the SAE data precomputation pipeline:
    1. Precompute SAE activations
    2. Precompute top activating images
    3. Precompute avg embeddings
    """
    
    # Construct data path
    data_path = config.model_embeddings_path
    
    # Step 1: Precompute SAE activations
    extract_sae_cmd = [
        "python", "extract_sae_embeddings.py",
        "-m", config.sae_model_weights_path,
        "-d", data_path,
        "-b", str(config.batch_size),
        "-o", config.output_dir,
        "--subset-size", str(config.subset_size)
    ]
    
    if "hsae" not in config.sae_model_weights_path.lower():
        try:
            subprocess.run(extract_sae_cmd, check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ Extract SAE embeddings failed with return code {e.returncode}")
            return
    
    # Step 3: Precompute top activating images
    utils_sae_cmd = ["python", "-m", "activations_preprocessing.utils_sae_activations"]
    
    try:
        subprocess.run(utils_sae_cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Utils SAE activations failed with return code {e.returncode}")
        return
    
    # Step 4: Precompute avg embeddings
    utils_embeddings_cmd = ["python", "-m", "activations_preprocessing.utils_embeddings"]
    
    try:
        subprocess.run(utils_embeddings_cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Utils embeddings failed with return code {e.returncode}")
        return
    
    print("✓ Pipeline completed successfully")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute SAE data precomputation pipeline"
    )
    # Placeholder for future argument parsing if needed
    # add argument subset_size
    parser.add_argument(
        "--subset_size", "-s",
        type=str,
        default="",
        help="Subset size for dataset (if applicable)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = PipelineConfig(
        batch_size=1024,
        sae_model_weights_path=DEFAULT_CONFIG.sae_model.weights_path,
        model_embeddings_path=DEFAULT_CONFIG.clip_embeddings_path,
        output_dir= PathBuilder().get_sae_activations_path(),
        subset_size=args.subset_size
    )
    run_precompute_pipeline(config)

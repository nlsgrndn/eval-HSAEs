"""
Execute calculation of SAE metrics.
Computes metrics for different SAE configurations (standard SAE, activation-based MSAE, architecture-based MSAE).
"""

import subprocess
from my_config import DEFAULT_CONFIG, Dataset

# Configuration
DATASET_NAME = "cc3m"
DATASET_SPLIT = "val"
BATCH_SIZE = 1024


def run_command(cmd, description):
    """Run a shell command."""
    print(f"Running: {description}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
    return result.returncode == 0


def compute_sae_metrics():
    """Compute metrics for each enabled SAE configuration."""
    print("Computing SAE metrics")
    
    config = DEFAULT_CONFIG

    # update dataset to val
    config.graph_eval_dataset = Dataset(name=DATASET_NAME, split=DATASET_SPLIT, subset_str="")

    model_path = config.sae_model.weights_path
    dataset_path = config.clip_embeddings_path
    
    cmd = [
        'python', 'extract_sae_embeddings.py',
        '-m', model_path,
        '-d', dataset_path,
        '-b', str(BATCH_SIZE),
        '-o', '.',
        '--no-save',
        '--save-metrics-csv'
    ]
    
    run_command(cmd, f"Computing metrics for {config.sae_model.simple_name}")


def main():
    """Main execution function."""
    compute_sae_metrics()
    print("Workflow completed")


if __name__ == "__main__":
    main()
from my_config import Dataset
from my_datasets.cc3m_dataset import CC3MLocalDataset
from my_datasets.imagenet1k_local import ImageNet1kLocalDataset
from torch.utils.data import Subset
from valuable_notebook_code_snippets import load_clip_embeddings_memmap, load_precomputed_sae_representations_memmap


def load_dataset(dataset_config: Dataset, return_file_path: bool):
    """Load dataset based on inference dataset string and split."""
    dataset_name_str = dataset_config.name
    split = dataset_config.split
    start_idx, end_idx = dataset_config.subset_start_and_end
    if dataset_name_str == "cc3m":
        dataset = CC3MLocalDataset(preprocess=None, split=split, return_file_path=return_file_path)
    elif dataset_name_str == "imagenet":
        dataset = ImageNet1kLocalDataset(preprocess=None, data_split=split, return_file_path=return_file_path)
    else:
        raise ValueError(f"Unknown inference dataset string: {dataset_name_str}")
    return Subset(dataset, range(start_idx, end_idx))

def load_embeddings_for_dataset(full_config, subset="graph_eval_dataset"):
    """Load embeddings for the specified dataset subset."""
    dataset_config = getattr(full_config, subset)
    embeddings_memmap = load_clip_embeddings_memmap(full_config.clip_embeddings_path)
    start_idx, end_idx = dataset_config.subset_start_and_end
    return embeddings_memmap[start_idx:end_idx]

def load_activations_for_dataset(full_config, subset="graph_eval_dataset"):
    """Load SAE activations for the specified dataset subset."""
    dataset_config = getattr(full_config, subset)
    activations_memmap = load_precomputed_sae_representations_memmap(full_config.sae_latents_path)
    start_idx, end_idx = dataset_config.subset_start_and_end
    return activations_memmap[start_idx:end_idx]
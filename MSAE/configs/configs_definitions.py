from dataclasses import dataclass
from typing import Optional
from configs.graph_configs import get_graph_name

@dataclass
class FoundationModel:
    name: str
    embedding_dim: int
    
    @property
    def simple_name(self) -> str:
        return f"{self.name[:3].lower()}"


GRAPH_SUBSET_SIZE = 500000
GRAPH_EVAL_SUBSET_SIZE = 500000
@dataclass
class Dataset:
    name: str
    split: str  # 'train' or 'val'
    subset_str: str
    # sae_activations_dataset_subset_size: int = 0 

    
    @property
    def size(self) -> int:
        sizes = {
            ("cc3m", "train"): 2820737,
            ("cc3m", "val"): 13002,
            ("imagenet", "train"): 1281167,
            ("imagenet", "val"): 50000,
        }
        return sizes.get((self.name, self.split), 0)
    
    @property
    def subset_start_and_end(self) -> tuple:
        if self.split == "train":
            if self.subset_str == "train":
                return (0, self.size - GRAPH_SUBSET_SIZE - GRAPH_EVAL_SUBSET_SIZE)
            elif self.subset_str == "inference":
                return (0, self.size - GRAPH_SUBSET_SIZE - GRAPH_EVAL_SUBSET_SIZE)
            elif self.subset_str == "graph_creation":
                return (self.size - GRAPH_SUBSET_SIZE - GRAPH_EVAL_SUBSET_SIZE, self.size - GRAPH_EVAL_SUBSET_SIZE)
            elif self.subset_str == "graph_evaluation":
                return (self.size - GRAPH_EVAL_SUBSET_SIZE, self.size)
            else:
                raise ValueError(f"Unknown subset_str: {self.subset_str}")
        elif self.split == "val":
            return (0, self.size)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    # @property
    # def sae_latents_size(self) -> int:
    #     return min(self.sae_activations_dataset_subset_size, self.size) if self.sae_activations_dataset_subset_size > 0 else self.size
    

    @property
    def simple_name(self) -> str:
        return f"{self.name}"

@dataclass
class SAEModel:
    name: str
    foundation_model: FoundationModel
    training_dataset: Dataset
    weights_folder: str = "./saved_models"
    width: int = 6144
    seed: Optional[int] = None

    @property
    def weights_path(self) -> str:
        return f"{self.weights_folder}/{self.name}.pth"

    @property
    def simple_name(self) -> str:
        if self.seed is not None:
            return f"{self.name}_seed{self.seed}"
        return f"{self.name}"

@dataclass
class Config:
    sae_model: SAEModel
    #inference_dataset: Dataset
    sae_filtering_strategy: str
    graph_creation_dataset: Dataset
    graph_eval_dataset: Dataset
    graph_name: str = get_graph_name()
    
    @property
    def sae_latents_path(self) -> str:
        ds = self.graph_eval_dataset
        fm = self.sae_model.foundation_model
        split_str = "validation" if ds.split == "val" else ds.split
        return (
            f"./sae_activations/{ds.name}_{fm.name}_{split_str}_image_{ds.size}_{fm.embedding_dim}_"
            f"{self.sae_model.name}h_repr_{ds.size}_{self.sae_model.width}.npy"
        )
    
    @property
    def clip_embeddings_path(self) -> str:
        ds = self.graph_eval_dataset
        fm = self.sae_model.foundation_model
        split_str = "validation" if ds.split == "val" else ds.split
        return f"./data/{ds.name}_{fm.name}_{split_str}_image_{ds.size}_{fm.embedding_dim}.npy"
    
    @property
    def simple_name(self) -> str:
        return f"{self.sae_model.simple_name}_{self.graph_eval_dataset.simple_name}_{self.sae_filtering_strategy}_{self.graph_name}"
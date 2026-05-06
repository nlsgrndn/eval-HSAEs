from dataclasses import dataclass


@dataclass
class HierarchicalityScoresConfig:
    top_k_embeddings: int = 20
    minimum_number_of_images_above_threshold: int = 20
    activation_threshold: float = 0.1
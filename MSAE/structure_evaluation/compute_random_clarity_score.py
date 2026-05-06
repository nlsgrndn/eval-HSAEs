import os

import torch

from label_assignment_strategies.individual_feature_scores.hierarchicality_scores_config import (
    HierarchicalityScoresConfig,
)
from my_config import DEFAULT_CONFIG
from my_utils import load_embeddings_for_dataset
from path_hub import PathBuilder

_PB = PathBuilder()


@torch.inference_mode()
def clarity_score(v):
    v_normed = torch.nn.functional.normalize(v, dim=-1)
    return ((v_normed.mean(-2).pow(2).sum(-1)) - 1 / v.shape[-2]) / (v.shape[-2] - 1) * v.shape[-2]


@torch.inference_mode()
def compute_random_clarity_scores(num_neurons_dummy: int, top_k: int, seed: int = 0) -> torch.Tensor:
    clip_embeddings = load_embeddings_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")
    clip_embeddings_tensor = torch.from_numpy(clip_embeddings.copy()).float()


    random_generator = torch.Generator().manual_seed(seed)
    random_indices = torch.randint(
        low=0,
        high=clip_embeddings_tensor.shape[0],
        size=(num_neurons_dummy, top_k),
        generator=random_generator,
    )
    sampled_embeddings = clip_embeddings_tensor[random_indices]
    return clarity_score(sampled_embeddings)


@torch.inference_mode()
def compute_random_pair_cosine_similarity_scores(num_pairs: int, seed: int = 0) -> torch.Tensor:
    clip_embeddings = load_embeddings_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")
    clip_embeddings_tensor = torch.from_numpy(clip_embeddings.copy()).float()

    random_generator = torch.Generator().manual_seed(seed)
    random_indices_a = torch.randint(
        low=0,
        high=clip_embeddings_tensor.shape[0],
        size=(num_pairs,),
        generator=random_generator,
    )
    random_indices_b = torch.randint(
        low=0,
        high=clip_embeddings_tensor.shape[0],
        size=(num_pairs,),
        generator=random_generator,
    )

    sampled_embeddings_a = torch.nn.functional.normalize(clip_embeddings_tensor[random_indices_a], dim=-1)
    sampled_embeddings_b = torch.nn.functional.normalize(clip_embeddings_tensor[random_indices_b], dim=-1)
    return (sampled_embeddings_a * sampled_embeddings_b).sum(dim=-1)

NUM_SAMPLES = 5000
if __name__ == "__main__":
    config = HierarchicalityScoresConfig()
    # random_clarity_scores = compute_random_clarity_scores(num_neurons_dummy=NUM_SAMPLES, top_k=config.top_k_embeddings, seed=0)
    # random_clarity_mean = random_clarity_scores.mean()
    # random_clarity_std = random_clarity_scores.std()
    random_pair_cosine_sim_scores = compute_random_pair_cosine_similarity_scores(num_pairs=NUM_SAMPLES, seed=0)
    random_pair_cosine_sim_mean = random_pair_cosine_sim_scores.mean()

    # folder = _PB.get_clarity_path()
    # os.makedirs(folder, exist_ok=True)
    # torch.save(random_clarity_scores, os.path.join(folder, "random_clarity_scores.pth"))

    # print(f"Random clarity score mean: {random_clarity_mean.item():.6f}")
    # print(f"Random clarity score std: {random_clarity_std.item():.6f}")
    print(f"Random pair cosine similarity mean: {random_pair_cosine_sim_mean.item():.6f}")
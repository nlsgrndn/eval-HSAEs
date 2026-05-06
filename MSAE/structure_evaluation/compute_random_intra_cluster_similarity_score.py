import numpy as np
import pandas as pd
from my_config import DEFAULT_CONFIG, CONFIGS
from my_utils import load_embeddings_for_dataset
from utils_sae_feature_properties import SAEDimensions

from sklearn.metrics.pairwise import cosine_similarity


def compute_random_avg_embeddings(num_features: int, top_k: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clip_embeddings = load_embeddings_for_dataset(DEFAULT_CONFIG, subset="graph_eval_dataset")
    random_indices = rng.integers(0, clip_embeddings.shape[0], size=(num_features, top_k))
    return clip_embeddings[random_indices].mean(axis=1)


NUM_RANDOM_FEATURES = 6144
TOP_K = 20
random_avg_embs = compute_random_avg_embeddings(NUM_RANDOM_FEATURES, top_k=TOP_K)
random_sim_matrix = cosine_similarity(random_avg_embs)
upper_tri = np.triu_indices_from(random_sim_matrix, k=1)
print(f"Random baseline (top_k={TOP_K}, n={NUM_RANDOM_FEATURES}): "
      f"mean={random_sim_matrix[upper_tri].mean():.4f}, "
      f"std={random_sim_matrix[upper_tri].std():.4f}")

for config in CONFIGS.values():
    print(config.simple_name[:10])
    avg_embeddings = (
        SAEDimensions(config=config).get_avg_embeddings_on_graph_eval_dataset()
    )  # shape (num_sae_dims, embedding_dim)
    # if "hsae" in config.simple_name:
    # import ipdb; ipdb.set_trace()   
    # check for nans
    if np.isnan(avg_embeddings).any():
        # print number of rows wth nans
        num_rows_with_nans = np.isnan(avg_embeddings).any(axis=1).sum()
        # print(f"Number of rows with nans: {num_rows_with_nans}")

        # remove nan rows
        avg_embeddings = avg_embeddings[~np.isnan(avg_embeddings).any(axis=1)]

    # print(avg_embeddings.shape)  # Compare across configs — is HSAE just smaller?
    # # Is the feature matrix low rank?
    # S = np.linalg.svd(avg_embeddings, compute_uv=False)
    # print(S[:10] / S.sum())  # Is variance concentrated in top few dims?

    similarity_matrix = cosine_similarity(avg_embeddings)

    # compute mean of upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    mean_similarity = similarity_matrix[upper_triangle_indices].mean()
    std_similarity = similarity_matrix[upper_triangle_indices].std()
    print(f"Mean similarity: {mean_similarity}")
    # print(f"Std similarity: {std_similarity}")
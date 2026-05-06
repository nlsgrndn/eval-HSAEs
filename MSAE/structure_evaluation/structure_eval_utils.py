from activations_preprocessing.act_behav_utils import (
    build_similarity_nonneg,
    zscore_columns,
)
from utils_sae_feature_properties import SAEDimensions
from my_config import DEFAULT_CONFIG


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MAX_NUM_ACTIVATIONS = 100000


def get_similarity_matrix(data_type, config = DEFAULT_CONFIG) -> np.ndarray:
    if data_type == "avg_embeddings":
        avg_embeddings = (
            SAEDimensions(config=config).get_avg_embeddings_on_graph_eval_dataset()
        )  # shape (num_sae_dims, embedding_dim)
        # replace nan with 0
        avg_embeddings = np.nan_to_num(avg_embeddings)
        similarity_matrix = cosine_similarity(
            avg_embeddings
        )  # shape (num_sae_dims, num_sae_dims)
        return similarity_matrix
    else:
        raise ValueError(f"Unknown data type: {data_type}")

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from label_assignment_strategies.labelling_and_scoring.constants_candidate_names import WORDS_AVG_EMB, WORDS_DEC_COL
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
from activations_preprocessing.utils_embeddings import load_avg_top_activation_images_embeddings
from valuable_notebook_code_snippets import (
    get_decoder_weights,
    get_dims_from_clip_embeddings_path,
    load_vocab_names,
)

_PB = PathBuilder()


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def _compute_top1_labels(
    normalized_targets: np.ndarray,
    normalized_vocab_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    num_latents = normalized_targets.shape[0]
    best_ids = np.empty(num_latents, dtype=np.int64)
    best_scores = np.empty(num_latents, dtype=np.float32)

    target_tensor = torch.from_numpy(normalized_targets)
    chunk_size = 256

    for start_idx in tqdm(range(0, num_latents, chunk_size), desc="Labelling SAE features"):
        end_idx = min(start_idx + chunk_size, num_latents)
        chunk_scores = target_tensor[start_idx:end_idx] @ normalized_vocab_t
        chunk_best_scores, chunk_best_ids = torch.max(chunk_scores, dim=1)
        best_ids[start_idx:end_idx] = chunk_best_ids.numpy()
        best_scores[start_idx:end_idx] = chunk_best_scores.numpy()

    return best_ids, best_scores

def run_labelling() -> tuple[str, str]:
    vocab_names = load_vocab_names("vocab/clip_disect_20k.txt")
    vocab_path = "vocab/embeddings_clip_ViT-L14_clip_disect_20k_20000_768.npy"
    dim_one, dim_two = get_dims_from_clip_embeddings_path(vocab_path)
    vocab_embeddings = np.memmap(vocab_path, dtype="float32", mode="r", shape=(dim_one, dim_two))

    normalized_vocab = _normalize_embeddings(np.asarray(vocab_embeddings, dtype=np.float32))
    normalized_vocab_t = torch.from_numpy(normalized_vocab).T

    decoder_weights = get_decoder_weights(DEFAULT_CONFIG.sae_model.weights_path)
    avg_embeddings = load_avg_top_activation_images_embeddings(DEFAULT_CONFIG)

    normalized_decoder_targets = _normalize_embeddings(np.asarray(decoder_weights, dtype=np.float32))
    normalized_avg_targets = _normalize_embeddings(np.asarray(avg_embeddings, dtype=np.float32))

    dec_ids, dec_scores = _compute_top1_labels(normalized_decoder_targets, normalized_vocab_t)
    avg_ids, avg_scores = _compute_top1_labels(normalized_avg_targets, normalized_vocab_t)

    num_latents = normalized_decoder_targets.shape[0]
    dec_col_labelling_df = pd.DataFrame(
        {
            "sae_id": np.arange(num_latents, dtype=np.int64),
            f"{WORDS_DEC_COL}_label": [vocab_names[idx] for idx in dec_ids],
            f"{WORDS_DEC_COL}_score": dec_scores,
            f"{WORDS_DEC_COL}_id_in_vocab": dec_ids,
        }
    )

    avg_emb_labelling_df = pd.DataFrame(
        {
            "sae_id": np.arange(num_latents, dtype=np.int64),
            f"{WORDS_AVG_EMB}_label": [vocab_names[idx] for idx in avg_ids],
            f"{WORDS_AVG_EMB}_score": avg_scores,
            f"{WORDS_AVG_EMB}_id_in_vocab": avg_ids,
        }
    )

    dec_col_labelling_df_path = os.path.join(_PB.get_labeling_output_path(), f"labelling_df_single_pass_{WORDS_DEC_COL}.csv")
    dec_col_labelling_df.to_csv(dec_col_labelling_df_path, index=False)
    print(f"Saved single-pass {WORDS_DEC_COL} labelling summary to {dec_col_labelling_df_path}")

    avg_emb_labelling_df_path = os.path.join(_PB.get_labeling_output_path(), f"labelling_df_single_pass_{WORDS_AVG_EMB}.csv")
    avg_emb_labelling_df.to_csv(avg_emb_labelling_df_path, index=False)
    print(f"Saved single-pass {WORDS_AVG_EMB} labelling summary to {avg_emb_labelling_df_path}")

    return dec_col_labelling_df_path, avg_emb_labelling_df_path


def main():
    run_labelling()


if __name__ == "__main__":
    main()

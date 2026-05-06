import os
import numpy as np
import pandas as pd
import torch

from path_hub import PathBuilder
from activations_preprocessing.utils_sae_activations import get_precomputed_top_k_path, load_precomputed_top_k
from my_config import DEFAULT_CONFIG
from activations_preprocessing.dependency_metrics import SaveAndLoad

_PB = PathBuilder()


def _append_ms_score_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    ms_score_path = os.path.join(_PB.get_monosemanticity_path(), "all_neurons_scores_approximate_i256_j1024MaxN500000.pth")
    ms_scores = torch.load(ms_score_path)
    summary_df["ms_score"] = summary_df["sae_id"].apply(lambda x: ms_scores[x].item())
    return summary_df


def _append_max_activation_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    _, _, max_activations = load_precomputed_top_k(get_precomputed_top_k_path())
    summary_df["max_activation"] = summary_df["sae_id"].apply(lambda x: max_activations[x].item())
    return summary_df


def _append_clarity_score_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    clarity_score_path = os.path.join(_PB.get_clarity_path(), "clarity_scores.pth")
    clarity_scores = torch.load(clarity_score_path)
    summary_df["clarity_score"] = summary_df["sae_id"].apply(lambda x: clarity_scores[x].item())
    return summary_df

def _append_dec_col_labelings_cols(summary_df: pd.DataFrame) -> pd.DataFrame:
    dec_col_labelling_path = os.path.join(_PB.get_labeling_output_path(), "labelling_df_single_pass_words_dec_col.csv")
    dec_col_labelling_df = pd.read_csv(dec_col_labelling_path)
    summary_df = summary_df.merge(
        dec_col_labelling_df[
            ["sae_id", "words_dec_col_label", "words_dec_col_score", "words_dec_col_id_in_vocab"]
            ],
        on="sae_id",
        how="left") 
    return summary_df

def _append_dummy_dec_col_labelings_cols(summary_df: pd.DataFrame) -> pd.DataFrame:
    dummy_labels_df = pd.DataFrame({
        "sae_id": summary_df["sae_id"],
        "words_dec_col_label": ["dummy"] * len(summary_df),
    })
    summary_df = summary_df.merge(
        dummy_labels_df,
        on="sae_id",
        how="left") 
    return summary_df


def _append_avg_emb_labeling_cols(summary_df: pd.DataFrame) -> pd.DataFrame:
    avg_emb_labelling_path = os.path.join(_PB.get_labeling_output_path(), "labelling_df_single_pass_words_avg_emb.csv")
    avg_emb_labelling_df = pd.read_csv(avg_emb_labelling_path)
    summary_df = summary_df.merge(
        avg_emb_labelling_df[
            ["sae_id", "words_avg_emb_label", "words_avg_emb_score", "words_avg_emb_id_in_vocab"]
            ],
        on="sae_id",
        how="left")
    return summary_df

def _append_feature_density_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    act_based_properties = SaveAndLoad.load_data_from_npy_arrays()
    # turn array into dataframe with columns sae_id and feature_density
    avg_emb_labelling_df = pd.DataFrame({
        "sae_id": np.arange(len(act_based_properties["base_rates"])),
        "feature_density": act_based_properties["base_rates"]
    })
    summary_df = summary_df.merge(
        avg_emb_labelling_df[
            ["sae_id", "feature_density"]
            ],
        on="sae_id",
        how="left")
    return summary_df

def create_summary_df() -> str:
    width = DEFAULT_CONFIG.sae_model.width
    summary_df = pd.DataFrame({"sae_id": np.arange(width, dtype=np.int64)})
    try:
        summary_df = _append_ms_score_column(summary_df)
    except FileNotFoundError as e:
        print(f"Warning: Could not append monosemanticity scores due to: {e}")

    summary_df = _append_max_activation_column(summary_df)

    try:
        summary_df = _append_clarity_score_column(summary_df)
    except FileNotFoundError as e:
        print(f"Warning: Could not append clarity scores due to: {e}")


    try:
        summary_df = _append_dec_col_labelings_cols(summary_df)
    except FileNotFoundError as e:
        print(f"Warning: Could not append words_dec_col labelling columns due to: {e}")
        summary_df = _append_dummy_dec_col_labelings_cols(summary_df)
        print("Filled words_dec_col labelling columns with dummy values.")

    try:
        summary_df = _append_avg_emb_labeling_cols(summary_df)
    except FileNotFoundError as e:
        print(f"Warning: Could not append words_avg_emb labelling columns due to: {e}")

    try:
        summary_df = _append_feature_density_column(summary_df)
    except FileNotFoundError as e:
        print(f"Warning: Could not append feature density column due to: {e}")

    summary_path = _PB.get_interpretability_data_path()
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved single-pass labelling summary to {summary_path}")
    return summary_path


def main():
    create_summary_df()


if __name__ == "__main__":
    main()

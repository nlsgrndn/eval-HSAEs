import pandas as pd
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
import os

def load_interpretability_data(config = DEFAULT_CONFIG):
    """Load the interpretability CSV data."""
    csv_path = PathBuilder(config=config).get_interpretability_data_path()
    return pd.read_csv(csv_path)

def load_sae_ids_with_labels(config=DEFAULT_CONFIG):
    df = load_interpretability_data(config)
    filtering_strategy = config.sae_filtering_strategy
    if filtering_strategy != "all":
        raise ValueError(
            f"Unsupported filtering strategy in simplified labelling pipeline: {filtering_strategy}. "
            "Use 'all'."
        )

    row_filter_condition = pd.Series([True] * len(df))
    df = df[row_filter_condition]
    return df["sae_id"].tolist(), df["words_dec_col_label"].tolist()


def load_all_sae_ids_with_dec_col_names(config=DEFAULT_CONFIG):
    df = load_interpretability_data(config)
    return df["sae_id"].tolist(), df["words_dec_col_label"].tolist()


def load_all_sae_ids_with_avg_emb_names(config=DEFAULT_CONFIG):
    df = load_interpretability_data(config)
    return df["sae_id"].tolist(), df["words_avg_emb_label"].tolist()
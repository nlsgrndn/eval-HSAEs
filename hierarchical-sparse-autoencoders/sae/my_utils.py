from dataclasses import dataclass

@dataclass
class Config:
    model_str: str
    data_split: str

import os
# assuming code is run from sae/ directory
BASE_PATH = "../../MSAE"
def get_embeddings_path(my_cfg):
    base_path = os.path.join(BASE_PATH, "data")
    # get input data
    if my_cfg.data_split == "train":
        split_str = "train_image_2820737_768"
    elif my_cfg.data_split == "val":
        split_str = "validation_image_13002_768"
    else:
        raise ValueError(f"Unknown data split: {my_cfg.data_split}")
    
    if my_cfg.model_str == "vit":
        model_str = "ViT-L~14"
    elif my_cfg.model_str == "dino":
        model_str = "dinov2-base"
    else:
        raise ValueError(f"Unknown model string: {my_cfg.model_str}")

    filename = f"cc3m_{model_str}_{split_str}.npy" # Example: "cc3m_ViT-L~14_validation_image_13002_768.npy"
    
    embeddings_path = os.path.join(base_path, filename)

    return embeddings_path

def get_output_paths(my_cfg, hsae_config_str):
    base_path = os.path.join(BASE_PATH, "sae_activations")

    # get input data
    if my_cfg.data_split == "train":
        split_str = "train_image_2820737_768"
    elif my_cfg.data_split == "val":
        split_str = "validation_image_13002_768"
    else:
        raise ValueError(f"Unknown data split: {my_cfg.data_split}")
    
    if my_cfg.model_str == "vit":
        model_str = "ViT-L~14"
        abbr_model_str = "vit"
    elif my_cfg.model_str == "dino":
        model_str = "dinov2-base"
        abbr_model_str = "dinov2"
    else:
        raise ValueError(f"Unknown model string: {my_cfg.model_str}")

    
    if hsae_config_str == "dino_361_16" or hsae_config_str == "vit_361_16": # OLD edge csae for backwards compatibility
        sae_model_str = f"hsae_361_16_6144_cc3m_{abbr_model_str}"
    else:
        sae_model_str = f"hsae_{hsae_config_str}_cc3m"

    data_size = split_str.split("_")[-2]
    repr_filename = f"cc3m_{model_str}_{split_str}_{sae_model_str}h_repr_{data_size}_6144.npy"
    output_filename = f"cc3m_{model_str}_{split_str}_{sae_model_str}h_output_{data_size}_768.npy"

    return os.path.join(base_path, repr_filename), os.path.join(base_path, output_filename)

import os
import numpy as np
def load_clip_embeddings_memmap(embeddings_path):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    # Split the embeddings path to determine dimensions
    embeddings_split = embeddings_path.rsplit(".")[-2].split("_")
    dim_one = int(embeddings_split[-2])  # e.g. 10000
    dim_two = int(embeddings_split[-1])  # e.g. 6144
    embeddings = np.memmap(embeddings_path, dtype='float32', mode='r', shape=(dim_one, dim_two))
    return embeddings
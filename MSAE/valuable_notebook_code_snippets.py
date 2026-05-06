import os
import torch
import numpy as np

from sae_model_loading_and_saving import load_model

def load_clip_embeddings(embeddings_path, max_datapoints = None):
    """
    Load precomputed CLIP embeddings from the specified path.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    # Split the embeddings path to determine dimensions
    embeddings_split = embeddings_path.rsplit(".")[-2].split("_")
    dim_one = int(embeddings_split[-2])  # e.g. 10000
    dim_two = int(embeddings_split[-1])  # e.g. 6144

    # Load the embeddings
    clip_embeddings = np.memmap(embeddings_path, dtype="float32", mode="r", shape=(dim_one, dim_two))

    if dim_one >= 100000 or dim_two >= 100000:
        if max_datapoints is not None and (max_datapoints <= 100000):
            clip_embeddings = clip_embeddings[:max_datapoints]
        else:
            raise ValueError(f"Dimensions too large: {dim_one} x {dim_two}. Embeddings should not be loaded with this function. Use load_precomputed_sae_representations_memmap instead.")
    return torch.from_numpy(np.copy(clip_embeddings))

def load_clip_embeddings_memmap(embeddings_path):
    """
    Load precomputed CLIP embeddings from the specified path.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")

    dim_one, dim_two = get_dims_from_clip_embeddings_path(embeddings_path)

    # Load the embeddings
    clip_embeddings = np.memmap(embeddings_path, dtype="float32", mode="r", shape=(dim_one, dim_two))
    return clip_embeddings

def get_dims_from_clip_embeddings_path(embeddings_path):
    # Split the embeddings path to determine dimensions
    embeddings_split = embeddings_path.rsplit(".")[-2].split("_")
    dim_one = int(embeddings_split[-2])  # e.g. 10000
    dim_two = int(embeddings_split[-1])  # e.g. 6144
    return dim_one, dim_two

def load_sae_model(weights_path):
    """
    Load the SAE model from the specified weights path.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    # Load the model
    sae_model, _, _, _= load_model(weights_path)
    return sae_model

def get_decoder_weights_and_bias(sae_model):
    """
    Extract decoder weights and bias from the SAE model.
    """
    if sae_model.tied:
        # use encoder weights as decoder weights
        decoder_weights = sae_model.encoder.data.T.clone().contiguous()
    else:
        decoder_weights = sae_model.decoder.data
    bias_pre = sae_model.pre_bias
    
    return decoder_weights, bias_pre

def get_decoder_weights(SAE_WEIGHTS_PATH):
    # if "hsae_vit_361_16_v2_cc3m" in SAE_WEIGHTS_PATH:
    #     # top_level_decoder_columns_vit_361_16_v2.npy
    #     # low_level_decoder_columns_vit_361_16_v2.npy
    #     top_level_decoder_columns = np.load("top_level_decoder_columns_vit_361_16_v2.npy") # shape (768, 361)
    #     low_level_decoder_columns = np.load("low_level_decoder_columns_vit_361_16_v2.npy") # shape (361, 768, 16)
    #     # goal shape: (6144, 768) by first using top level, then concatenating the flattened low level columns for each of 361 top level latents
    #     # then add zero columns for the remaining 6144 - 361*16 + 361
    #     decoder_weights = np.zeros((6144, 768), dtype=np.float32)
    #     decoder_weights[:361, :] = top_level_decoder_columns.T
    #     for i in range(361):
    #         decoder_weights[361 + i*16 : 361 + (i+1)*16, :] = low_level_decoder_columns[i, :, :].T
    #     import ipdb; ipdb.set_trace()
    #     return decoder_weights
    sae_model = load_sae_model(SAE_WEIGHTS_PATH)
    decoder_weights, decoder_bias = get_decoder_weights_and_bias(sae_model)
    decoder_weights = decoder_weights.cpu().numpy()
    return decoder_weights

def get_encoder_weights(sae_weights_path):
    sae_model = load_sae_model(sae_weights_path)
    encoder_weights = sae_model.encoder.data # [n_inputs, n_latents]
    encoder_weights = encoder_weights.cpu().numpy()
    return encoder_weights
    

def load_precomputed_sae_representations(embeddings_path, max_datapoints = None):
    """
    Load precomputed SAE representations from the specified path.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    # Split the embeddings path to determine dimensions
    embeddings_split = embeddings_path.rsplit(".")[-2].split("_")
    dim_one = int(embeddings_split[-2])  # e.g. 10000
    dim_two = int(embeddings_split[-1])  # e.g. 6144
    
    # Load the representations
    sae_representations = np.memmap(embeddings_path, dtype="float32", mode="r", 
                                    shape=(dim_one, dim_two))
    
    if dim_one >= 100000 or dim_two >= 100000:
        if max_datapoints is not None and (max_datapoints <= 100000):
            sae_representations = sae_representations[:max_datapoints]
        else:
            raise ValueError(f"Dimensions too large: {dim_one} x {dim_two}. Embeddings should not be loaded with this function. Use load_precomputed_sae_representations_memmap instead.")

    return torch.from_numpy(np.copy(sae_representations))

def load_precomputed_sae_representations_memmap(embeddings_path, shape=None):
    """
    Load precomputed SAE representations from the specified path.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    if shape is not None:
        dim_one, dim_two = shape
    else:
        # Split the embeddings path to determine dimensions
        embeddings_split = embeddings_path.rsplit(".")[-2].split("_")
        dim_one = int(embeddings_split[-2])  # e.g. 10000
        dim_two = int(embeddings_split[-1])  # e.g. 6144
    
    # Load the representations
    sae_representations = np.memmap(embeddings_path, dtype="float32", mode="r", 
                                    shape=(dim_one, dim_two))
    return sae_representations

def load_vocab_names(vocab_file):
    """
    Load vocabulary names from the specified file.
    """
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")
    
    with open(vocab_file, 'r') as f:
        vocab_names = [line.strip() for line in f.readlines()]
    return vocab_names


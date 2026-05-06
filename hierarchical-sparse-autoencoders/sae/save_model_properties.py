from load_model import load_model, load_config, parse_args
from run_moe_eqx_utils import load_clip_embeddings_memmap
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from collections import defaultdict

def save_decoder_columns(model, filename):
    decoder_columns = model.top_level_autoencoder.decoder # Shape: (input_dim, num_experts)
    decoder_columns_np = np.array(decoder_columns)  # Convert to NumPy array
    np.save(filename, decoder_columns_np)
    print(f"Saved decoder columns to {filename}")

def low_level_decoder_columns(model, filename):
    decoder_columns_low_dim = model.decoder_weights # Shape: (num_experts, subspace_dim, atoms_per_subspace)
    # project each decoder column up from subspace_dim to input_dim using the corresponding W_Up
    W_up = model.W_up  # Shape: (num_experts, input_dim, subspace_dim)
    decoder_columns = jnp.einsum('eij,ejk->eik', W_up, decoder_columns_low_dim)
    decoder_columns_np = np.array(decoder_columns)  # Convert to NumPy array
    np.save(filename, decoder_columns_np)
    print(f"Saved low-level decoder columns to {filename}")

if __name__ == "__main__":

    args = parse_args()
    config_file_name = args.config_file
    model_checkpoint_path = args.model_checkpoint_path
    name = args.name
    input_dim = args.input_dim
    restore_from = f"{model_checkpoint_path}/{name}"

    config_dict = load_config(config_file_name)

    model = load_model(config_dict, restore_from, input_dim)

    DECODER_COLUMNS_DIR = "./decoder_columns"
    import os
    os.makedirs(DECODER_COLUMNS_DIR, exist_ok=True)

    save_decoder_columns(model, f'{DECODER_COLUMNS_DIR}/top_level_decoder_columns_{name}.npy')
    low_level_decoder_columns(model, f'{DECODER_COLUMNS_DIR}/low_level_decoder_columns_{name}.npy')


from run_moe_eqx_utils import get_restore_vals, get_model, load_clip_embeddings_memmap
import json
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import reduce
from operator import getitem
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    

def load_model(config_dict, restore_from, input_dim):
    subspace_dim = config_dict['subspace_dim']
    num_experts = config_dict['num_experts']
    atoms_per_subspace = config_dict['atoms_per_subspace']
    k = config_dict['k']
    bias = config_dict['bias']
    save_checkpoints = config_dict['save_checkpoints']
    restore_step = config_dict['restore_step']

    mngr, restored = get_restore_vals(save_checkpoints, restore_from, restore_step)
    print(f"Read restore from {restore_from} at step {restore_step}")

    key = jax.random.PRNGKey(0)
    model, hyperparameters = get_model(input_dim, subspace_dim, atoms_per_subspace, num_experts, k, bias, key)
    hyperparameters = restored.hyperparameters
    model = restore_state(model,restored.model)
    return model

def restore_state(model, restored):
    restored = jax.tree.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, restored)
    
    # Handle top_level_autoencoder separately if it exists
    if "top_level_autoencoder" in restored and hasattr(model, "top_level_autoencoder"):
        tla_restored = restored["top_level_autoencoder"]
        tla_updates = {}
        for sub_key, sub_value in tla_restored.items():
            if hasattr(model.top_level_autoencoder, sub_key):
                tla_updates[sub_key] = sub_value
        
        if tla_updates:
            updated_tla = eqx.tree_at(
                lambda x: [getattr(x, k) for k in tla_updates.keys()],
                model.top_level_autoencoder,
                list(tla_updates.values()),
                is_leaf=lambda x: x is None
            )
            
            model = eqx.tree_at(
                lambda x: x.top_level_autoencoder,
                model,
                updated_tla,
                is_leaf=lambda x: x is None
            )
        
        # Remove top_level_autoencoder from restored to avoid double processing
        restored = {k: v for k, v in restored.items() if k != "top_level_autoencoder"}
    
    # Update remaining attributes
    if restored:
        remaining_updates = {}
        for key, value in restored.items():
            if hasattr(model, key):
                remaining_updates[key] = value
        
        if remaining_updates:
            model = eqx.tree_at(
                lambda x: [getattr(x, k) for k in remaining_updates.keys()],
                model,
                list(remaining_updates.values()),
                is_leaf=lambda x: x is None
            )
    
    return model

# def load_config_values(config):
#     # Load configuration
#     batch_size = config['batch_size']
#     subspace_dim = config['subspace_dim']
#     num_experts = config['num_experts']
#     atoms_per_subspace = config['atoms_per_subspace']
#     k = config['k']
#     l1_penalty = config['l1_penalty']
#     ortho_penalty = config['ortho_penalty']
#     num_epochs = config['num_epochs']
#     wandb_run_name = config['wandb_run_name']
#     num_epochs = config['num_epochs']
#     lr_peak = config['lr_peak']
#     lr_init = config['lr_init']
#     norm_clip = config['norm_clip']
#     warmup_steps = config['warmup_steps']
#     #restore_from = #config['restore_from']
#     restore_from = f"{model_checkpoint_path}/{name}"
#     bias = config['bias']
#     save_checkpoints = config['save_checkpoints']
#     restore_step = config['restore_step']
#     fsdp_shard = config['fsdp_shard']
#     return num_experts, atoms_per_subspace

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to the config file')
    parser.add_argument('--model_checkpoint_path', type=str, default="/home/ngrandien/thesis/hierarchical-sparse-autoencoders/checkpoints", help='Path to the model checkpoints')
    parser.add_argument('--name', type=str, help='Name of the model checkpoint directory')
    parser.add_argument('--input_dim', type=int, default=768, help='Input dimension of the model')
    # Optionally add more arguments as needed, e.g. for output paths, etc.
    parser.add_argument('--use_val', action='store_true', help='Whether to use the validation set instead of the training set for loading embeddings')
    args = parser.parse_args()
    return args

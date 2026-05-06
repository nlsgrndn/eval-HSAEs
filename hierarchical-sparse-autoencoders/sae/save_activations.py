from load_model import load_model, load_config, parse_args
from run_moe_eqx_utils import load_clip_embeddings_memmap
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from collections import defaultdict
from moe_eqx import mask_codes
from tqdm import tqdm
import os
from my_sae_dataset import SAEDataset
from my_utils import Config, get_embeddings_path, get_output_paths

def save_activations_memmap(latents_list, filename):
    # story as numpy memmap
    latents_jnp = jnp.concatenate(latents_list, axis=0)
    latents_np = jnp.array(latents_jnp)
    memmap = np.memmap(filename, dtype='float32', mode='w+', shape=latents_np.shape)
    # Write data to memmap
    memmap[:] = np.array(latents_np)
    memmap.flush()

def encode(model, batched_embeddings):
    batched_embeddings = jnp.asarray(batched_embeddings, dtype=jnp.float32)
    top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = jax.vmap(model.encode)(batched_embeddings)
    return top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values


def build_batch_activations(
    masked_top_level_latent_codes,
    masked_expert_specific_codes,
    batch_top_k_indices,
    total_width
):
    batch_size = masked_top_level_latent_codes.shape[0]
    num_experts = masked_top_level_latent_codes.shape[1]
    atoms_per_subspace = masked_expert_specific_codes.shape[2]
    # total_width = num_experts + num_experts * atoms_per_subspace
    batch_activations = jnp.zeros((batch_size, total_width), dtype=masked_top_level_latent_codes.dtype)
    batch_activations = batch_activations.at[:, :num_experts].set(masked_top_level_latent_codes)

    # Gather the selected expert subspace codes for each datapoint and scatter them
    # into their flattened positions in the activation vector.
    selected_codes = masked_expert_specific_codes[
        jnp.arange(batch_size)[:, None],
        batch_top_k_indices,
        :,
    ]

    atom_offsets = jnp.arange(atoms_per_subspace)
    flat_col_indices = (
        num_experts
        + batch_top_k_indices[:, :, None] * atoms_per_subspace
        + atom_offsets[None, None, :]
    ).reshape(batch_size, -1)

    flat_values = selected_codes.reshape(batch_size, -1)
    row_indices = jnp.arange(batch_size)[:, None]
    batch_activations = batch_activations.at[row_indices, flat_col_indices].set(flat_values)
    return batch_activations

if __name__ == "__main__":
    args = parse_args()
    config_file_name = args.config_file
    model_checkpoint_path = args.model_checkpoint_path
    name = args.name
    input_dim = args.input_dim

    restore_from = f"{model_checkpoint_path}/{name}"

    config_dict = load_config(config_file_name)

    model = load_model(config_dict, restore_from, input_dim)


    my_cfg = Config(
        model_str=config_dict['model_str'],
        data_split=config_dict['data_split'] if not args.use_val else "val"
    )
    embeddings_path = get_embeddings_path(my_cfg)
    embeddings = load_clip_embeddings_memmap(embeddings_path)

    

    num_experts = config_dict['num_experts']
    atoms_per_subspace = config_dict['atoms_per_subspace']
    TOTAL_ACTIVATIONS = embeddings.shape[0]
    BATCH_SIZE = 1024

    sae_dataset = SAEDataset(embeddings_path, mean_center=True)

    # create output directory and memmap file
    result_shape = (TOTAL_ACTIVATIONS, 6144) #hardcoded for now WITH TRAILING ZEROS, since we know the config of the model we want to extract from
    #result_shape = (TOTAL_ACTIVATIONS, NUM_EXPERTS + NUM_EXPERTS * atoms_per_subspace)

    acts_filename, rec_filename = get_output_paths(my_cfg, name)
    acts_memmap = np.memmap(acts_filename, dtype='float32', mode='w+', shape=result_shape)
    rec_memmap = np.memmap(rec_filename, dtype='float32', mode='w+', shape=embeddings.shape)

    num_batches = (TOTAL_ACTIVATIONS + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches), total=num_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, TOTAL_ACTIVATIONS)
        batch_embeddings = sae_dataset.process_data(sae_dataset.data[start:end])
        batch_top_level_latent_codes, batch_expert_specific_codes, batch_top_k_indices, batch_top_k_values = encode(model, batch_embeddings)
        batch_masked_top_level_latent_codes, batch_masked_expert_specific_codes = mask_codes(
            batch_top_level_latent_codes, batch_expert_specific_codes, batch_top_k_indices, batch_top_k_values,
                )
        x_hat, decoder_norms, x_hat_top = jax.vmap(model.decode)(
            batch_masked_expert_specific_codes, batch_top_k_indices, batch_top_k_values, 
            )
        
        # OLD non-optimized code
        # batch_activations = batch_activations.at[:, :NUM_EXPERTS].set(masked_top_level_latent_codes)
        # for datapoint in range(BATCH_SIZE):
        #     for id in batch_top_k_indices[datapoint]:
        #         start_idx = NUM_EXPERTS + id * atoms_per_subspace
        #         end_idx = start_idx + atoms_per_subspace
        #         batch_activations = batch_activations.at[datapoint, start_idx:end_idx].set(masked_expert_specific_codes[datapoint, id])
        batch_activations = build_batch_activations(
            batch_masked_top_level_latent_codes,
            batch_masked_expert_specific_codes,
            batch_top_k_indices,
            result_shape[1])

        # write to memmap
        acts_memmap[start:end] = np.asarray(batch_activations, dtype=np.float32)
        acts_memmap.flush()

        # Scale back up to original norm and add mean back.
        x_hat = sae_dataset.unprocess_data(np.asarray(x_hat, dtype=np.float32))

        rec_memmap[start:end] = x_hat
        rec_memmap.flush()
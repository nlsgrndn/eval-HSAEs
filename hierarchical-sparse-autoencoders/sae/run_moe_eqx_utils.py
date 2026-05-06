import json
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
import optax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, Mesh, PartitionSpec, NamedSharding
import functools
import tensorstore as ts
import wandb
import collections
import itertools
import numpy as np
#import orbax
import orbax.checkpoint as ocp
from operator import getitem
from functools import reduce
from my_sae_dataset import SAEDataset

from moe_eqx import (
    MixtureOfExperts_v2,
    tensorstore_data_generator,
    train_step,
    eps
)

from my_utils import load_clip_embeddings_memmap

# # where is your shuffled dataset of embeddings? e.g. '/net/scratch2/muchane/gemma_shuffle'
# data_path = ""
# def get_dataloader(epochs, batch_size):
#     dataset = ts.open(
#         {
#         'driver': 'zarr3',
#         'cache_pool': {'total_bytes_limit': 10E9},
#         'recheck_cached_data': 'open',
#         'kvstore': {
#             'driver': 'file',
#             'file_io_concurrency': {'limit': 1024},
#             'path': data_path,
#             },
#         },
#         dtype=ts.float32,
#         chunk_layout=ts.ChunkLayout(
#         write_chunk_shape=[102400, 2304],
#         ),
#         shape=[988364800, 2304],
#     ).result()

#     train_loader = tensorstore_data_generator(dataset, batch_size=batch_size, epochs=epochs)
#     example_batch = next(train_loader)
#     input_dim = example_batch[0].shape[0]

#     return train_loader, example_batch, input_dim

# def tensorstore_data_generator(dataset,batch_size=32512,chunks_per=127, epochs=1):
#     outer_loop_i = 9652 // chunks_per
#     inner_loop_i = (102400*chunks_per) // batch_size
#     reads = [dataset[i*(102400*chunks_per):(i+1)*(102400*chunks_per)] for i in range(outer_loop_i)]
#     curr_read = reads[0].read()
#     for _ in range(epochs):
#         for i in range(outer_loop_i):
#             data_arr = curr_read.result()
#             if i != (outer_loop_i -1):
#                 curr_read = reads[i+1].read()
#             for j in range(inner_loop_i):
#                 yield data_arr[j::inner_loop_i][:batch_size]


def get_datagenerator(embeddings_memmap, batch_size, epochs):
    """
    Generate batches of embeddings for training.
    
    Args:
        embeddings_memmap: Memory-mapped numpy array of embeddings
        batch_size: Size of each batch
        epochs: Number of epochs to iterate through the data
        
    Yields:
        Batches of embeddings as JAX arrays
    """
    num_samples = embeddings_memmap.shape[0]
    
    for epoch in range(epochs):
        # Generate batches sequentially
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            # Load batch from memmap and convert to JAX array
            batch = embeddings_memmap[i:end_idx]
            yield to_jax_batch(batch)


def get_dataloader(epochs, batch_size, embeddings_path):
    DATASET_SPLIT_SIZE = 1820737
    embeddings_memmap = load_clip_embeddings_memmap(embeddings_path)[:DATASET_SPLIT_SIZE] # only use this split for training
    return get_datagenerator(embeddings_memmap, batch_size, epochs), None, 768, DATASET_SPLIT_SIZE

def to_jax_batch(batch):
    """Convert array-like batch data to JAX float32 at the loader boundary."""
    return jnp.asarray(np.asarray(batch, dtype=np.float32))


def get_datagenerator_v2(sae_dataset, batch_size, epochs, rng):
    num_samples = len(sae_dataset)
    for _ in range(epochs):
        shuffled_indices = rng.permutation(num_samples)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = shuffled_indices[start_idx:end_idx]
            batch = sae_dataset.data[batch_indices]
            processed_batch = sae_dataset.process_data(batch)
            yield to_jax_batch(processed_batch)

def get_dataloader_v2(epochs, batch_size, embeddings_path, seed):
    DATASET_SPLIT_SIZE = 1820737
    sae_dataset = SAEDataset(embeddings_path, mean_center=True, max_size=DATASET_SPLIT_SIZE)
    rng = np.random.default_rng(seed)
    return get_datagenerator_v2(sae_dataset, batch_size, epochs, rng), None, 768, DATASET_SPLIT_SIZE

def get_model(input_dim, subspace_dim, atoms_per_subspace, num_experts, k, bias, key):
    hyperparameters = {
        "input_dim": input_dim,
        "subspace_dim": subspace_dim,
        "atoms_per_subspace": atoms_per_subspace,
        "num_experts": num_experts,
        "k": k,
        "use_bias": bias
    }
    model = MixtureOfExperts_v2(
        **hyperparameters,
        key=key
    )
    return model, hyperparameters

def get_lr_schedule(lr_init, lr_peak, warmup_steps, num_steps, batch_size):
    def cosine_decay_schedule(init_value, peak_value, warmup_steps, decay_steps, alpha=0.0):
        if warmup_steps == 0:
            return optax.cosine_decay_schedule(
                init_value=init_value,
                decay_steps=decay_steps,
                alpha=alpha)
        warmup_fn = optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps
        )
        cosine_decay_fn = optax.cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=decay_steps,
            alpha=alpha
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_decay_fn],
            boundaries=[warmup_steps]
        )
    learning_rate_fn = cosine_decay_schedule(
        init_value=lr_init,
        peak_value=lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=num_steps - warmup_steps,
        alpha=0.1
    )

    return learning_rate_fn

def get_optimizer(learning_rate_fn, norm_clip):
    optimizer = optax.chain(
        optax.clip_by_global_norm(norm_clip),
        optax.adam(learning_rate_fn, b1=0.9)
    )
    return optimizer

def regularizer_warmup_fn(warmup_steps):
    @jax.jit
    def curr_weight(step):
        return jnp.minimum(1.0, step/warmup_steps)
    return curr_weight

def prefetch_to_sharding(iterator, size: int, sharding):
    queue = collections.deque()

    def _prefetch(xs):
        arr = jax.device_put(xs, sharding)
        #arr = arr - jnp.mean(arr, axis=1, keepdims=True)
        norms = jnp.linalg.norm(arr, axis=1, keepdims=True)
        return arr/(norms+eps)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)
@jax.jit
def update_expert_tracker(tracker, top_k_indices, expert_specific_codes):
    tracker_arr = jnp.repeat(jnp.expand_dims(jnp.zeros_like(tracker), axis=0), top_k_indices.shape[0], axis=0)
    tracker_arr = tracker + tracker_arr.at[jnp.arange(tracker_arr.shape[0])[:, None], top_k_indices].set(expert_specific_codes > 0).sum(axis = 0)
    return tracker_arr

def get_restore_vals(save_checkpoints, path, step):
    options = ocp.CheckpointManagerOptions(max_to_keep=save_checkpoints)
    mngr = ocp.CheckpointManager(
        path, options=options, item_names=('opt_state', 'model', 'hyperparameters')
    )
    if step:
        return mngr, mngr.restore(step,args=ocp.args.Composite(model=ocp.args.StandardRestore(),hyperparameters=ocp.args.JsonRestore(),opt_state=ocp.args.StandardRestore()))
    return mngr, mngr.restore(mngr.latest_step(),args=ocp.args.Composite(model=ocp.args.StandardRestore(),hyperparameters=ocp.args.JsonRestore(),opt_state=ocp.args.StandardRestore()))

def restore_state(state, restored):
    restored = jax.tree.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, restored)
    return jax.tree_util.tree_map_with_path(lambda p, _: reduce(getitem, [list(x.__dict__.values())[0] for x in p], restored), state)
# NOTE: this is sketchy and questionably efficient
def shard_model(model, num_experts):
    mesh = Mesh(mesh_utils.create_device_mesh((jax.local_device_count())), ('0'))
    P = PartitionSpec
    def shard_layer(layer,sharded_layer_size):
        if not eqx.is_array(layer):
            return layer
        if sharded_layer_size in layer.shape:
            axis_num = int(jnp.argwhere(jnp.asarray(layer.shape)==sharded_layer_size).reshape(-1)[0])
            return jax.device_put(layer, NamedSharding(mesh, P(*([None]*axis_num), '0')))
        else:
            return jax.device_put(layer, NamedSharding(mesh, P()))

    shard_fn = functools.partial(shard_layer, sharded_layer_size=num_experts)
    return jax.tree.map(shard_fn,model)

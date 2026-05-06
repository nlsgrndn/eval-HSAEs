import os
# Set this BEFORE importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import json
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import wandb
import orbax.checkpoint as ocp
import tqdm as tqdm

from moe_eqx import train_step

from run_moe_eqx_utils import (
    get_dataloader,
    get_dataloader_v2,
    get_model,
    get_lr_schedule,
    get_optimizer,
    regularizer_warmup_fn,
    prefetch_to_sharding,
    update_expert_tracker,
    get_restore_vals,
    restore_state,
    shard_model
)

from my_utils import Config, get_embeddings_path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

if len(sys.argv) != 2:
    print("Usage: python run_moe_eqx.py <config_file_path>")
    sys.exit(1)

config = load_config(sys.argv[1])

# Load configuration
batch_size = config['batch_size']
subspace_dim = config['subspace_dim']
num_experts = config['num_experts']
atoms_per_subspace = config['atoms_per_subspace']
k = config['k']
l1_penalty = config['l1_penalty']
ortho_penalty = config['ortho_penalty']
num_epochs = config['num_epochs']
wandb_run_name = config['wandb_run_name']
num_epochs = config['num_epochs']
lr_peak = config['lr_peak']
lr_init = config['lr_init']
norm_clip = config['norm_clip']
warmup_steps = config['warmup_steps']
restore_from = config['restore_from']
bias = config['bias']
save_checkpoints = config['save_checkpoints']
restore_step = config['restore_step']
fsdp_shard = config['fsdp_shard']
seed = config['seed']

# Where do you want model weight checkpoint to be stored? e.g. /net/projects2/interp/moe_ocp
checkpoint_path = os.path.abspath("./checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)
with open(f'{checkpoint_path}/{wandb_run_name}.json', 'w') as config_copy:
    json.dump(config, config_copy, indent=4)

print("config loaded, initializing dataloader...")

local_cfg = Config(
    model_str=config['model_str'],
    data_split=config['data_split']
)
embeddings_path = get_embeddings_path(local_cfg)
train_loader, example_batch, input_dim, dataset_size = get_dataloader_v2(num_epochs, batch_size, embeddings_path, seed)

if restore_from:
    mngr, restored = get_restore_vals(save_checkpoints, restore_from, restore_step)
    print(f"Read restore from {restore_from} at step {restore_step}")

key = jax.random.PRNGKey(seed)
model, hyperparameters = get_model(input_dim, subspace_dim, atoms_per_subspace, num_experts, k, bias, key)
if restore_from:
    hyperparameters = restored.hyperparameters
    model = restore_state(model,restored.model)

if fsdp_shard:
    model = shard_model(model, num_experts)
print("Model loaded...")

#import tensorflow as tf
#tf.config.set_visible_devices([], device_type='GPU')
#g = jnp.load('/net/projects/veitch/geometry_llms/unembeddings/gemma-2-2b/clean_unembeddings.npy')
#g = g * jnp.sqrt(g.shape[0] / g.shape[1]) # set the norms to be close to 1
#
#def create_train_loader(data, batch_size, shuffle_buffer_size=1000):
#    """
#    Creates a train loader using tf.data that yields batches from a large data array,
#    with shuffling and repeating.
#
#    Args:
#        data: The training data array of shape [number_samples, sample_dimension].
#        batch_size: The desired batch size.
#        shuffle_buffer_size: The size of the shuffle buffer (default: 1000).
#
#    Returns:
#        A tf.data.Dataset object that yields batches of data.
#    """
#    dataset = tf.data.Dataset.from_tensor_slices(data)
#    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)  # Shuffle the data
#    dataset = dataset.batch(batch_size, drop_remainder=True)  # Create batches
#    dataset = dataset.repeat()  # Repeat the dataset indefinitely
#    return dataset
#
#
#
#def prepare_tf_data(xs):
#  """Convert a input batch to numpy arrays."""
#  def _prepare(x):
#    # Use _numpy() for zero-copy conversion between TF and NumPy.
#    x = x._numpy()
#    return x
#  return jax.tree_util.tree_map(_prepare, xs)
#
#ds = create_train_loader(g, 8192, shuffle_buffer_size=g.shape[0])
#train_loader = map(prepare_tf_data, ds)

dataset_size = dataset_size
num_steps = (dataset_size // batch_size) * num_epochs - 1
# num_steps_fixed = (dataset_size // batch_size) * 2 - 1

epoch_30_step = (dataset_size // batch_size) * 30

learning_rate_fn = get_lr_schedule(lr_init, lr_peak, warmup_steps, num_steps, batch_size)
optimizer = get_optimizer(learning_rate_fn, norm_clip)


opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
if restore_from:
    opt_state = restore_state(opt_state,restored.opt_state)


curr_weight_fn = regularizer_warmup_fn(warmup_steps)

num_devices = len(jax.devices())
# sharding = PositionalSharding(mesh_utils.create_device_mesh((num_devices,1)))
train_loader_pf = train_loader # prefetch_to_sharding(train_loader, 2, sharding)

run = wandb.init(project="moe_testing", name=wandb_run_name, config=config)

latent_tracker = jnp.zeros((num_experts, atoms_per_subspace))

if not restore_from:
    path = ocp.test_utils.erase_and_create_empty(f'{checkpoint_path}/{wandb_run_name}')
    options = ocp.CheckpointManagerOptions(max_to_keep=save_checkpoints)
    mngr = ocp.CheckpointManager(
        path, options=options, item_names=('opt_state', 'model', 'hyperparameters')
    )



print("Starting training...")
for step in range(num_steps):
    if restore_step:
        step += restore_step
    batch = next(train_loader_pf)
    if step == 0:
        print(
            f"First batch type: {type(batch)}, shape: {getattr(batch, 'shape', None)}, "
            f"dtype: {getattr(batch, 'dtype', None)}"
        )
        assert getattr(batch, "ndim", None) == 2, "Expected batch to have shape [batch, dim]"
    frac = curr_weight_fn(step)
    model, opt_state, loss, aux_out = train_step(model, batch, opt_state, l1_penalty*frac, ortho_penalty*frac, optimizer)
    stats_dict, top_k_indices, top_k_values, top_level_latent_codes, expert_specific_codes = aux_out
    latent_tracker = update_expert_tracker(latent_tracker, top_k_indices, expert_specific_codes)
    if step % 100 == 0 or step == epoch_30_step:
        stats_dict["avg_nonzero"] = jnp.sum(top_k_values >0) / batch_size
        stats_dict["step"] = step
        stats_dict["epoch"] = step // (dataset_size // batch_size)
        stats_dict["dead_atoms"] = jnp.sum(latent_tracker == 0)
        stats_dict["dead_experts"] = jnp.sum(jnp.sum(latent_tracker,axis=-1) == 0)
        run.log(stats_dict)
        if step % 500 == 0 or step == epoch_30_step:
            latent_tracker = jnp.zeros((num_experts, atoms_per_subspace))
            print(f"Step {step}, Loss: {loss:.4f}")
        if step % 1500 == 0 or step == epoch_30_step:
            mngr.save(
                step,
                args=ocp.args.Composite(
                    opt_state=ocp.args.StandardSave(opt_state),
                    model=ocp.args.StandardSave(model),
                    hyperparameters=ocp.args.JsonSave(hyperparameters),
                ),
            )

mngr.save(
    step,
    args=ocp.args.Composite(
        opt_state=ocp.args.StandardSave(opt_state),
        model=ocp.args.StandardSave(model),
        hyperparameters=ocp.args.JsonSave(hyperparameters),
    ),
)
mngr.wait_until_finished()

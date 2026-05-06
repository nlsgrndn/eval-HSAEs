import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple
import functools

eps = 1e-6

def naive_top_k(data, k):
    """Top k implementation built with argmax.
    Faster for smaller k."""

    def top_1(data):
        indice = jnp.argmax(data, axis=1)
        value = jax.vmap(lambda x, y: x[y])(data, indice)
        data = jax.vmap(lambda x, y: x.at[y].set(-jnp.inf))(data, indice)
        return data, value, indice

    def scannable_top_1(carry, unused):
        data = carry
        data, value, indice = top_1(data)
        return data, (value, indice)

    data, (values, indices) = jax.lax.scan(scannable_top_1, data, (), k)

    return values.T.reshape(-1), indices.T.reshape(-1)
def leaky_offset_relu(x, negative_slope=1e-2, offset=0.):
    return jnp.where(x >= offset, x, negative_slope * x)

def cross_orthogonality_penalty(encoder, decoder):
  E = encoder.T
  D = decoder.T

  E = E / (jnp.linalg.norm(E, axis=0, keepdims=True) + eps)
  D = D / (jnp.linalg.norm(D, axis=1, keepdims=True) + eps)
  ED = D @ E
  dim = ED.shape[0]
  return jnp.linalg.norm(ED - jnp.diag(ED)) / (dim**2 - dim)

def gram_matrix_regularizer(weights):
    weights = weights / (jnp.linalg.norm(weights, axis=0, keepdims=True) + 1e-6)
    gram_matrix = jnp.dot(weights.T, weights)
    off_diagonal_elements = gram_matrix - jnp.diag(jnp.diag(gram_matrix))
    dim = off_diagonal_elements.shape[0]
    regularization_penalty = jnp.sum(off_diagonal_elements ** 2) / (dim**2 - dim)
    return weights.shape[0] * regularization_penalty

def general_ortho_regularizer(weights, other):
    weights = weights / (jnp.linalg.norm(weights, axis=0, keepdims=True) + eps)
    other = other / (jnp.linalg.norm(other, axis=0, keepdims=True) + eps)
    gram_matrix = jnp.dot(other, weights)
    dim = gram_matrix.shape[0]
    regularization_penalty = jnp.sum(gram_matrix ** 2) / (dim**2 - dim)
    return weights.shape[0] * regularization_penalty

class Autoencoder(eqx.Module):
    encoder: jnp.ndarray
    decoder: jnp.ndarray
    bias: jnp.ndarray
    use_bias: bool
    offset: float

    def __init__(self, latent_dim: int, input_dim: int, use_bias: bool = True, key=None):
        initializer = jax.nn.initializers.he_uniform(in_axis=-1,out_axis=-2)
        self.encoder = initializer(key, (latent_dim, input_dim), jnp.float32)
        self.decoder = self.encoder.T
        self.bias = jnp.zeros(input_dim) if use_bias else None
        self.use_bias = use_bias
        self.offset = 1.0/jnp.sqrt(input_dim)

    def encode(self, x):
        x = x - self.bias if self.use_bias else x
        codes = self.encoder @ x
        #return codes
        #return jnp.where(codes >= self.offset, codes, 0)
        #return jax.nn.relu(codes)
        return leaky_offset_relu(codes, negative_slope=0., offset=self.offset)

    def decode(self, z):
        return jnp.dot(z, self.decoder) + (self.bias if self.use_bias else 0)

    def top_k_decode(self, top_k_indices, top_k_values):
        decoder_weights = self.get_decoder()
        # top_k_indices is now 1D after vmap, so we don't need [:, :, None]
        selected_decoder_weights = decoder_weights[:, top_k_indices]
        return selected_decoder_weights @ top_k_values + (self.bias if self.use_bias else 0)

    def get_decoder(self):
        return self.decoder# / jnp.linalg.norm(self.decoder, axis=0, keepdims=True)

    def get_encoder(self):
        return self.encoder

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

class MixtureOfExperts_v2(eqx.Module):
    input_dim: int
    subspace_dim: int
    atoms_per_subspace: int
    num_experts: int
    k: int
    top_level_autoencoder: Autoencoder
    W_down: jnp.ndarray
    W_up: jnp.ndarray
    encoder_weights: jnp.ndarray
    decoder_weights: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, input_dim: int, subspace_dim: int, atoms_per_subspace: int,
                 num_experts: int, k: int, use_bias: bool = False, key=None):
        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.atoms_per_subspace = atoms_per_subspace
        self.num_experts = num_experts
        self.k = k

        keys = jax.random.split(key, 3)
        # Correct initialization of top_level_autoencoder
        self.top_level_autoencoder = Autoencoder(num_experts, input_dim, use_bias=False, key=keys[0])

        self.bias = jnp.load("/net/projects2/interp/gemma2-2B_sample_geom_median.npy") if use_bias else None
        self.bias = jnp.zeros_like(self.bias) if use_bias else None

        initializer = jax.nn.initializers.he_uniform(in_axis=-1,out_axis=(-3,-2))

        self.W_down = initializer(keys[1], (num_experts, subspace_dim, input_dim), jnp.float32)
        self.W_down /= jnp.linalg.norm(self.W_down, axis=-1, keepdims=True)
        self.W_up = jnp.transpose(self.W_down, (0, 2, 1))

        self.encoder_weights = initializer(keys[2], (num_experts, atoms_per_subspace, subspace_dim))
        self.encoder_weights /= jnp.linalg.norm(self.encoder_weights, axis=-1, keepdims=True)
        self.decoder_weights = jnp.transpose(self.encoder_weights, (0, 2, 1))

    def encode(self, x):
        x = x - self.bias if self.bias is not None else x
        top_level_latent_codes = self.top_level_autoencoder.encode(x)
        top_k_values, top_k_indices = naive_top_k(top_level_latent_codes[None, :], self.k)

        selected_W_down = self.W_down[top_k_indices]
        selected_encoder_weights = self.encoder_weights[top_k_indices]

        subspace_outputs = selected_W_down @ x
        expert_specific_codes = jax.vmap(lambda x, y: x @ y)(selected_encoder_weights, subspace_outputs)
        offset_val = 1.0/jnp.sqrt(self.input_dim)
        expert_specific_codes = leaky_offset_relu(expert_specific_codes, offset=offset_val, negative_slope=1e-2)

        return top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values

    def decode(self, expert_specific_codes, top_k_indices, top_k_values):
        top_level_output = self.top_level_autoencoder.top_k_decode(top_k_indices, top_k_values)
        norms = jnp.linalg.norm(self.decoder_weights, axis=1, keepdims=True)
        decoder_weights = self.decoder_weights# / (norms + eps)
        Wup = self.W_up# / (jnp.linalg.norm(self.W_up, axis=1, keepdims=True) + eps)

        selected_decoder_weights = decoder_weights[top_k_indices]
        selected_W_up = Wup[top_k_indices]

        decoded_subspace_outputs = jax.vmap(lambda x, y: x @ y)(selected_decoder_weights, expert_specific_codes)
        reconstructed_inputs = jnp.sum(jax.vmap(lambda x, y: x @ y)(selected_W_up, decoded_subspace_outputs),axis=0)

        final_output = reconstructed_inputs + top_level_output
        final_output = final_output + (self.bias if self.bias is not None else 0)

        return final_output, norms, (top_level_output + (self.bias if self.bias is not None else 0))

    def get_top_level_decoder(self):
        return self.top_level_autoencoder.get_decoder()

    def get_top_level_encoder(self):
        return self.top_level_autoencoder.get_encoder()

    def __call__(self, x):
        top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = self.encode(x)
        x_hat, _ = self.decode(expert_specific_codes, top_k_indices, top_k_values)
        return x_hat

def mask_codes(top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values):
    mask = jnp.zeros_like(top_level_latent_codes)
    mask_vals = jnp.where(top_k_values > 0, 1, 0)
    mask = mask.at[jnp.arange(mask.shape[0])[:, None], top_k_indices].set(mask_vals)
    masked_top_level_latent_codes = mask * top_level_latent_codes

    max_code_indices = jnp.argmax(expert_specific_codes, axis=-1)
    low_mask = jnp.zeros_like(expert_specific_codes)
    batch_indices = jnp.arange(low_mask.shape[0])[:, None]
    expert_indices = jnp.arange(low_mask.shape[1])[None, :]
    low_mask = low_mask.at[batch_indices, expert_indices, max_code_indices].set(1)
    masked_expert_specific_codes = low_mask * expert_specific_codes

    submask = jnp.take_along_axis(mask, top_k_indices, axis=1)
    masked_expert_specific_codes = submask[:, :, None] * masked_expert_specific_codes

    return masked_top_level_latent_codes, masked_expert_specific_codes

# THIS CAUSES AN ERROR @eqx.filter_jit
@functools.partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(model: MixtureOfExperts_v2, batch: jnp.ndarray, l1_penalty: float, ortho_penalty: float) -> Tuple[float, Tuple]:
    top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = jax.vmap(model.encode)(batch)
    _, masked_expert_specific_codes = mask_codes(top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values)

    x_hat, decoder_norms, x_hat_top = jax.vmap(model.decode)(masked_expert_specific_codes, top_k_indices, top_k_values)

    reconstruction_loss = jnp.mean(jnp.sum(jnp.square(batch - x_hat), axis=-1))
    reconstruction_loss_top = jnp.mean(jnp.sum(jnp.square(batch - x_hat_top), axis=-1))

    batch_size = expert_specific_codes.shape[0]
    l1_loss = l1_penalty * (jnp.sum(jnp.abs(top_level_latent_codes)) + jnp.sum(jnp.abs(expert_specific_codes))) / batch_size

    top_ortho_loss = 0#0.5*gram_matrix_regularizer(model.get_top_level_encoder().T) + 0.5*gram_matrix_regularizer(model.get_top_level_decoder())
    low_ortho_loss = 0#0.5*jnp.mean(jax.vmap(gram_matrix_regularizer)(jnp.transpose(model.encoder_weights, (0, 2, 1)))) + 0.5*jnp.mean(jax.vmap(gram_matrix_regularizer)(model.decoder_weights))
    #OLD: proj_ortho_loss = 0.5*gram_matrix_regularizer(jnp.transpose(model.W_down, (2, 0, 1)).reshape(model.input_dim, -1)) + 0.5*gram_matrix_regularizer(jnp.transpose(model.W_up, (1, 0, 2)).reshape(model.input_dim, -1))
    top_bottom_ortho_loss = 0#jnp.mean(jax.vmap(general_ortho_regularizer)(model.W_up, model.get_top_level_decoder().T))
    cross_ortho_loss = cross_orthogonality_penalty(model.get_top_level_encoder(), model.get_top_level_decoder())
    ortho_loss = cross_ortho_loss #top_ortho_loss + low_ortho_loss + top_bottom_ortho_loss
    #nonzero_loss = 1E-1*(jnp.minimum(8/(jnp.sum(top_k_values >0) / batch_size), 1.0))

    total_loss = reconstruction_loss + l1_loss + ortho_penalty * ortho_loss + 1E-1*reconstruction_loss_top
    # + nonzero_loss
    stats_dict = {
        "reconstruction_loss": reconstruction_loss,
        "total_l1_loss": l1_loss,
        "total_ortho_loss": ortho_penalty * ortho_loss,
        "top_ortho_loss": top_ortho_loss,
        "low_ortho_loss": low_ortho_loss,
        "top_bottom_ortho_loss": top_bottom_ortho_loss,
        "loss": total_loss,
        "decoder_max_norm": jnp.max(decoder_norms),
        "decoder_min_norm": jnp.min(decoder_norms)
    }
    return total_loss, (stats_dict, top_k_indices, top_k_values, top_level_latent_codes, expert_specific_codes)

def project_away_grads(grads, model):
    def vector_reject(a, b):
        normed_b = b / jnp.linalg.norm(b)
        return a - (jnp.dot(a, normed_b)) * normed_b

    # Handle top level decoder
    top_where = lambda t: t.top_level_autoencoder.decoder
    top_decoder_mat = model.top_level_autoencoder.decoder
    normalized_top_decoder = top_decoder_mat / jnp.linalg.norm(top_decoder_mat, axis=0, keepdims=True)
    top_replace_fn = lambda dec: jax.vmap(vector_reject, in_axes=1, out_axes=1)(dec, normalized_top_decoder)
    new_grads = eqx.tree_at(where=top_where, pytree=grads, replace_fn=top_replace_fn)

    # Handle expert decoders
    expert_decoder_mat = model.decoder_weights
    normalized_expert_decoder = expert_decoder_mat / jnp.linalg.norm(expert_decoder_mat, axis=1, keepdims=True)
    expert_where = lambda t: t.decoder_weights
    expert_replace_fn = lambda dec: jax.vmap(jax.vmap(vector_reject, in_axes=1, out_axes=1))(dec, normalized_expert_decoder)
    return eqx.tree_at(where=expert_where, pytree=new_grads, replace_fn=expert_replace_fn)

def normalize_decoders(model):
    # Normalize top level decoder
    top_where = lambda t: t.top_level_autoencoder.decoder
    top_replace_fn = lambda dec: dec / jnp.linalg.norm(dec, axis=0, keepdims=True)
    model = eqx.tree_at(where=top_where, pytree=model, replace_fn=top_replace_fn)

    # Normalize expert decoders
    expert_where = lambda t: t.decoder_weights
    expert_replace_fn = lambda dec: dec / jnp.linalg.norm(dec, axis=1, keepdims=True)
    return eqx.tree_at(where=expert_where, pytree=model, replace_fn=expert_replace_fn)

def update_model(model, grads, opt_state, optimizer):
    # First project the gradients
    projected_grads = project_away_grads(grads, model)
    # Then do the optimizer step with projected gradients
    updates, new_opt_state = optimizer.update(projected_grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    # Finally normalize both decoders
    new_model = normalize_decoders(new_model)
    return new_model, new_opt_state

@eqx.filter_jit
def train_step(model: MixtureOfExperts_v2, batch: jnp.ndarray, opt_state, l1_penalty: float, ortho_penalty: float, optimizer) -> Tuple[MixtureOfExperts_v2, optax.OptState, float, Tuple]:
    (loss, aux_out), grads = loss_fn(model, batch, l1_penalty, ortho_penalty)
    model, opt_state = update_model(model, grads, opt_state, optimizer)
    return model, opt_state, loss, aux_out

def tensorstore_data_generator(dataset,batch_size=32512,chunks_per=127, epochs=1):
    outer_loop_i = 9652 // chunks_per
    inner_loop_i = (102400*chunks_per) // batch_size
    reads = [dataset[i*(102400*chunks_per):(i+1)*(102400*chunks_per)] for i in range(outer_loop_i)]
    curr_read = reads[0].read()
    for _ in range(epochs):
        for i in range(outer_loop_i):
            data_arr = curr_read.result()
            if i != (outer_loop_i -1):
                curr_read = reads[i+1].read()
            for j in range(inner_loop_i):
                yield data_arr[j::inner_loop_i][:batch_size]

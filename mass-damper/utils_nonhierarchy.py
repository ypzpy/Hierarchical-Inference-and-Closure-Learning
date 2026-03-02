import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
import jax.scipy.stats as stats
from jax import jit, vmap, pmap, grad, value_and_grad, lax
from jax.scipy.signal import convolve
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
import yaml
from types import SimpleNamespace


def convolve_avg(array, window):
    kernel = jnp.ones(window)
    new_array = convolve(array, kernel, mode='same') / convolve(jnp.ones_like(array), kernel, mode='same')
    return new_array
    
activation_functions = {
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'silu':nn.silu,
    'elu':nn.elu,
    'sin':jnp.sin
    # Add more activation functions as needed
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'axes.labelsize':   16,
    'axes.titlesize':   16,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
          })
# latex font definition
plt.rc('legend',fontsize=14)
plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
    
def array_to_batch_list(data, size_batch=128):
    '''
        function that takes in [n_data_pts, dim_data] and splits
        it into a list of batches of size size_batch with the last
        batch containg the rest of the data.
    '''
    dataset = []
    num_whole_batches = int( data.shape[0] / size_batch)
    for i in range( num_whole_batches ):
        dataset.append( data[i * size_batch: (i+1) * size_batch] )
    if data.shape[0] % size_batch != 0: 
        dataset.append(data[num_whole_batches * size_batch:])

    print('num_batches = ', len(dataset))
    n = data.shape[0] - num_whole_batches * size_batch
    if n >= 0:
        print(' items in uneven batch: ', n)
    else: print('data.shape[0] < size_batch')

    return dataset


def get_keys_and_rng(key, num=1):
    '''
        pass rng and n
        get keys and rng
    '''
    key, rng = jax.random.split(key)
    keys = jax.random.split(key, num=num)
    return keys, rng


def get_exp_sequence(lower, upper, n):
    return jnp.exp(jnp.linspace(jnp.log(lower), jnp.log(upper), n))
def get_linear_sequence(lower, upper, n):
    return jnp.linspace(lower, upper, n)


# @partial(jax.jit, static_argnames='dim')
def fill_lower_tri(v, dim):
    '''
        Make vector into lower-triangular matrix 
        https://github.com/google/jax/discussions/10146
    '''
    idxL = jnp.tril_indices(dim)
    L    = jnp.zeros((dim, dim), dtype=v.dtype).at[idxL].set(v) 
    idxD = jnp.diag_indices(dim)
    LD   = nn.softplus( L.at[idxD].get() )
    L    = L.at[idxD].set(LD)
    return L


def load_config(path='config.yml'):
    """Loads configuration from a YAML file."""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # Recursively convert dict to SimpleNamespace for dot notation access (e.g., cfg.training.epochs)
    def dict_to_sns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
        return d
    return dict_to_sns(config_dict)


def single_chain_initialisation(rng_key, cfg):
    """
    Initializes the state for a single Langevin chain, including both
    system parameters (theta) for all N systems and the hyperparameters (phi).
    """
    
    rng_key, theta_key, phi_key = random.split(rng_key, 3)
    
    # Initialize theta for N systems
    k_key, z0_key, zdot_key = random.split(theta_key, 3)
    k_params = jnp.log(cfg.prior.mean.k_mean) + random.normal(k_key, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.prior.variance.k_variance)
    z0_params = cfg.prior.mean.z0_mean + random.normal(z0_key, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.prior.variance.z0_variance)
    zdot_params = cfg.prior.mean.zdot_mean + random.normal(zdot_key, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.prior.variance.zdot_variance)

    theta = jnp.stack([k_params, z0_params, zdot_params], axis=-1)
    
    return theta


def vmap_single_chain_initialisation(rng_key, cfg):
    
    batched_rng_key = jax.random.split(rng_key, cfg.langevin_sampler.n_chains)
    init_fn = partial(single_chain_initialisation, cfg=cfg)
    return jax.vmap(init_fn)(batched_rng_key)


def generate_parameter_set(cfg,rng_key):
    log_k_mean = jnp.log(cfg.true_parameters.hyperprior_k)
    z0_mean = cfg.true_parameters.hyperprior_z0
    zdot_mean = cfg.true_parameters.hyperprior_zdot

    key_k, key_z0, key_zdot = jax.random.split(rng_key, 3)

    k_samples = jax.random.normal(key_k, (cfg.data_systems.n_systems)) * jnp.sqrt(cfg.true_parameters.tau_phi_theta) + log_k_mean
    z0_samples = jax.random.normal(key_z0, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_z0) + z0_mean
    zdot_samples = jax.random.normal(key_zdot, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_zdot) + zdot_mean

    parameter_matrix = jnp.stack([k_samples, z0_samples, zdot_samples], axis=-1)

    return parameter_matrix
import jax
import jax.numpy as jnp
from utils import *
from constant_FNO_supervised import * 
from jax.scipy.sparse.linalg import cg as jax_cg
from jaxopt import FixedPointIteration

def nonlinear_function(x):
    
    return 7 * (sigmoid_fn(x*3) - 0.5)


def sigmoid_fn(x):
    return 1 / (1 + jnp.exp(-x))


def G_burgers_true(parameters):
    """
    Solves the modified Burgers setup based on the provided equation image.
    Equation: u_t + c(u) * u_x = nu * u_xx
    Where:
        c(u) = u^2 (Nonlinear convection speed)
        nu = parameters[0] (Constant viscosity to be inferred)
    
    Args:
        parameters: (2,) array containing [nu, amplitude]
    """ 

    substeps = 10
    dt = ht / substeps

    # --- 2. Parse Parameters ---
    nu = jnp.exp(parameters[0])        # Viscosity (Constant)
    amplitude = parameters[1]          # Amplitude for IC

    # --- 3. Initial Condition ---
    w = 1.0 
    u_init = amplitude * jnp.sin(2 * jnp.pi * w * x) * jnp.sin(jnp.pi * x)

    # --- 4. Physics Step ---
    def physics_step(i, u):
        u_left = jnp.roll(u, 1)
        u_right = jnp.roll(u, -1)
        
        wave_speed = 7 * (sigmoid_fn(u*3) - 0.5)
        
        du_dx_backward = (u - u_left) / hx
        du_dx_forward = (u_right - u) / hx
        
        convection = jnp.where(u > 0, 
                               wave_speed * du_dx_backward, 
                               wave_speed * du_dx_forward)
        
        u_xx = (u_right - 2*u + u_left) / (hx**2)
        diffusion = nu * u_xx
        
        u_new = u + dt * (diffusion - convection)
        
        # Boundary Conditions
        # u_new = u_new.at[0].set(0.0)
        # u_new = u_new.at[-1].set(0.0)
        
        return u_new

    # --- 5. Scan Loop ---
    def scan_body(carry, _):
        u_current = carry
        u_next_save = jax.lax.fori_loop(0, substeps, physics_step, u_current)
        return u_next_save, u_next_save

    _, u_history_raw = jax.lax.scan(scan_body, u_init, None, length=nt - 1)
    
    u_full = jnp.concatenate([u_init[None, :], u_history_raw], axis=0)
    
    return u_full


@jit
def vmap_batched_burgers_true(batched_parameters):
    """
    parameters: (batch, 3)
    returns: (batch, nx, ny) PDE solutions
    """
    return vmap(G_burgers_true)(batched_parameters)


def generate_padded_observation_matrices(key, num_systems, nt, nx, 
                                         n_x_obs=10, 
                                         min_t_obs=5, 
                                         max_t_obs=10):
    """
    Generate H matrices where:
    1. X-locations are UNIFORM and FIXED.
    2. T-locations are RANDOM (Unsorted).
    3. Custom distribution: 10 (80%), 8 (10%), 6 (10%).
    """
    N_total = nt * nx
    
    max_rows = max_t_obs * n_x_obs
    
    x_indices_float = jnp.linspace(0, nx - 1, n_x_obs)
    x_indices = jnp.round(x_indices_float).astype(jnp.int32)
    
    possible_values = jnp.array([6, 8, 10], dtype=jnp.int32)
    
    probs = jnp.array([0.1, 0.1, 0.8])
    
    keys = jax.random.split(key, num_systems)

    def _generate_single_system(k):
        k_count, k_time = jax.random.split(k, 2)
        
        n_valid_t = jax.random.choice(k_count, possible_values, shape=(), p=probs)
        
        t_indices = jax.random.choice(k_time, nt, shape=(max_t_obs,), replace=False)
        
        time_mask = jnp.arange(max_t_obs) < n_valid_t
        full_mask = jnp.repeat(time_mask, n_x_obs)
        
        flat_indices_grid = t_indices[:, None] * nx + x_indices[None, :]
        idx_flat = flat_indices_grid.flatten()
        
        H = jnp.zeros((max_rows, N_total))
        row_indices = jnp.arange(max_rows)
        H = H.at[row_indices, idx_flat].set(1.0)
        
        H = H * full_mask[:, None]
        
        return H, full_mask

    H_matrices, masks = jax.vmap(_generate_single_system)(keys)
    return H_matrices, masks

def obtain_observations(parameter_matrix, rng_key):
    """
    Generate noisy observations with padding handled.
    """
    num_systems = cfg.data_systems.n_systems
    noise_std = cfg.data_systems.obser_noise
    nt, nx = cfg.data_systems.nt, cfg.data_systems.nx
    
    n_x_obs = 10       
    
    min_t = 6            
    max_t = 10           
    
    sim_data = jax.vmap(G_burgers_true)(parameter_matrix)
    u_flat = sim_data.reshape(num_systems, -1)
    
    key_H, key_noise = jax.random.split(rng_key)
    H_mats, masks = generate_padded_observation_matrices(
        key_H, num_systems, nt, nx, 
        n_x_obs, min_t, max_t
    )
    
    true_obs = jnp.einsum('bij, bj -> bi', H_mats, u_flat)
    
    raw_noise = jax.random.normal(key_noise, shape=true_obs.shape)
    masked_noise = raw_noise * masks * noise_std
    
    obs_data = true_obs + masked_noise
    
    return H_mats, masks, obs_data


def single_chain_initialisation(rng_key):
    
    single_chain = jnp.zeros((cfg.data_systems.n_systems+2)*cfg.data_systems.no_parameters)

    rng_key,key1,key2,key3,key4,key5,key6 = jax.random.split(rng_key,7)
        
    log_z1_mu = jax.random.normal(key1)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z1)+jnp.log(cfg.hyperprior.mean.hyperprior_z1_prior)
    z2_mu = jax.random.normal(key2)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z2)+cfg.hyperprior.mean.hyperprior_z2_prior

    z1_tau = jnp.exp(jax.random.normal(key4)*jnp.sqrt(cfg.hyperprior.variance.hyperprior_variance_tau_z1)+jnp.log(cfg.hyperprior.variance.hyperprior_variance_mu_z1))
    z2_tau = jnp.exp(jax.random.normal(key5)*jnp.sqrt(cfg.hyperprior.variance.hyperprior_variance_tau_z2)+jnp.log(cfg.hyperprior.variance.hyperprior_variance_mu_z2))

    
    for i in range(cfg.data_systems.n_systems):
        
        key,key1,key2,rng_key = jax.random.split(rng_key,4)
        single_chain = single_chain.at[i*cfg.data_systems.no_parameters].set(jax.random.normal(key)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z1)+jnp.log(cfg.hyperprior.mean.hyperprior_z1_prior))
        single_chain = single_chain.at[i*cfg.data_systems.no_parameters+1].set(jax.random.normal(key1)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z2)+cfg.hyperprior.mean.hyperprior_z2_prior)

    single_chain = single_chain.at[-2*cfg.data_systems.no_parameters:-cfg.data_systems.no_parameters].set(jnp.array([log_z1_mu,z2_mu]))
    single_chain = single_chain.at[-cfg.data_systems.no_parameters:].set(jnp.log(jnp.array([z1_tau,z2_tau])))

    return single_chain


def vmap_single_chain_initialisation(rng_key):
    
    batched_rng_key = jax.random.split(rng_key, cfg.langevin_sampler.n_chains)
    # init_fn = partial(single_chain_initialisation, cfg=cfg)
    
    return vmap(single_chain_initialisation)(batched_rng_key)


def generate_parameter_set(rng_key):
    log_z1_mean = jnp.log(cfg.true_parameters.hyperprior_z1)
    z2_mean = cfg.true_parameters.hyperprior_z2

    key_z1, key_z2 = jax.random.split(rng_key, 2)

    z1_samples = jax.random.normal(key_z1, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_z1) + log_z1_mean
    z2_samples = jax.random.normal(key_z2, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_z2) + z2_mean

    parameter_matrix = jnp.stack([z1_samples, z2_samples], axis=-1)
    # parameter_matrix = jnp.stack([z1_samples, z2_samples], axis=-1)

    return parameter_matrix
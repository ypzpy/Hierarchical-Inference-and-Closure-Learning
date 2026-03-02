import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
from fno import *
from utils import *
from data_generation import *
from constant_FNO_supervised import *
from langevin_FNO import *

    
def supervised_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn):
    # --- Forward simulations using the FNO model ---
    batch = theta_batch.shape[0]
    inputs = jnp.tile(theta_batch[:, None, None, :], (1, nt, nx, 1))
    inputs = jnp.concatenate([inputs, grid_tiled[:8]], axis=-1)
    
    out = beta_apply_fn(params_beta, inputs).squeeze(-1)
    
    batched_zs = jnp.mean(inputs[:, :, :, 0:cfg.data_systems.no_parameters],axis=(1,2))
    
    # Stitch: (Batch, 1, nx) + (Batch, nt-1, nx) -> (Batch, nt, nx)
    pred = out

    GTs = vmap_batched_burgers_train(batched_zs, params_alpha, alpha_apply_fn)
    loss = jnp.mean((pred - GTs) ** 2)

    return loss


@partial(jax.jit, static_argnames=('alpha_apply_fn',))
def G_burgers_train(parameters, params_alpha, alpha_apply_fn):
    """
    Solves the Burgers equation for training.
    Logic strictly mirrors G_burgers_true, assuming c(u) > 0.
    
    Equation: u_t + c(u) * u_x = nu * u_xx
    """
    
    substeps = 10  
    dt = ht / substeps

    # --- 2. Parse Parameters ---
    nu = jnp.exp(parameters[0])       
    amplitude = parameters[1]         
    
    # --- 3. Initial Condition ---
    w = 1.0 
    u_init = amplitude * jnp.sin(2 * jnp.pi * w * x) * jnp.sin(jnp.pi * x)
    
    def physics_step(i, u):
        
        # === A. Learned Convection Speed c(u) ===
        u_in = u.reshape(-1, 1)
        out_abs = alpha_apply_fn({'params': params_alpha}, jnp.abs(u_in))
        alpha_out = jnp.sign(u_in) * out_abs
        # alpha_out = alpha_apply_fn({'params': params_alpha}, u_in)
        c_u = alpha_out.squeeze(-1)
        
        # === B. Convection Term: c(u) * u_x (Upwind Scheme) ===
        u_left = jnp.roll(u, 1)
        u_right = jnp.roll(u, -1)
        
        du_dx_backward = (u - u_left) / hx
        du_dx_forward = (u_right - u) / hx
        
        convection = jnp.where(u > 0, 
                               c_u * du_dx_backward, 
                               c_u * du_dx_forward)
        
        # === C. Diffusion Term ===
        u_xx = (u_right - 2*u + u_left) / (hx**2)
        diffusion = nu * u_xx
        
        # === D. Explicit Update ===
        u_new = u + dt * (diffusion - convection)
        
        # Boundary Conditions (Dirichlet)
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


@partial(jax.jit, static_argnames=('alpha_apply_fn', ))
def vmap_batched_burgers_train(batched_parameters, params_alpha, alpha_apply_fn):
    """
    parameters: (batch, 3)
    returns: (batch, nx, ny) PDE solutions
    """
    return vmap(G_burgers_train, in_axes=(0, None, None))(batched_parameters, params_alpha, alpha_apply_fn)


def vmap_batch_mc_expectation(batched_parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn):
    
    return vmap(log_posterior,in_axes=(0,None,None,None,None,None))(batched_parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn).mean()


def alpha_loss_function(parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn):

    return - vmap_batch_mc_expectation(parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn)
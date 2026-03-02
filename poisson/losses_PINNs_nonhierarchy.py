import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax, jacfwd
from jax.tree_util import tree_map
from fno import *
from utils import *
from data_generation import *
from constant_PINNs_nonhierarchy import *
from langevin_PINNs_nonhierarchy import *


def pinn_physics_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn, rng_key=None):
    
    B = theta_batch.shape[0]
    N_domain = 5000 
    N_bc = 1000     
    
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    if rng_key is not None:
        key_domain, key_bc = jax.random.split(rng_key)
        xy_domain = jax.random.uniform(key_domain, (B, N_domain, 2), minval=0.0, maxval=1.0)
    else:
        key_bc = None
        xy_domain = jnp.zeros((B, N_domain, 2))
        
    def get_a_value_single(u_val, x, y, z_coeffs):
        pi = jnp.pi
        phi_1 = jnp.sin(2 * pi * x) * jnp.sin(2 * pi * y)
        phi_2 = jnp.sin(2 * pi * x) * jnp.sin(pi * y)
        phi_3 = jnp.sin(pi * x) * jnp.sin(2 * pi * y)
        basis_vals = jnp.stack([phi_1, phi_2, phi_3])
        Z_val = jnp.dot(z_coeffs, basis_vals)
        Z_field = nn.softplus(Z_val)
        g_u = alpha_apply_fn(params_alpha, u_val.reshape(1, 1))[0, 0]
        return Z_field * sigmoid_fn(g_u)

    def pde_residual_single_point(xy, z_coeffs):
        x, y = xy[0], xy[1]
        def u_fn(_xy): 
            return beta_apply_fn(params_beta, _xy, z_coeffs)
        
        def flux_fn(_xy):
            _x, _y = _xy[0], _xy[1]
            u_val, u_grad = jax.value_and_grad(u_fn)(_xy)
            a_val = get_a_value_single(u_val, _x, _y, z_coeffs)
            return a_val * u_grad

        div_flux = jnp.trace(jacfwd(flux_fn)(xy))
        s_val = 2 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
        return (-div_flux - s_val)**2

    batch_pde_residuals = vmap(vmap(pde_residual_single_point, in_axes=(0, None)), in_axes=(0, 0))(xy_domain, theta_batch)
    loss_pde = jnp.mean(batch_pde_residuals)

    def bc_loss_single_system(theta, key):
        if key is None:
            xy_bc_all = jnp.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
        else:
            k1, k2, k3, k4 = jax.random.split(key, 4)
            
            # Bottom: (x, 0)
            x_b = jax.random.uniform(k1, (N_bc,), minval=x_min, maxval=x_max)
            y_b = jnp.full((N_bc,), y_min)
            bc_bottom = jnp.stack([x_b, y_b], axis=1)

            # Top: (x, 1)
            x_t = jax.random.uniform(k2, (N_bc,), minval=x_min, maxval=x_max)
            y_t = jnp.full((N_bc,), y_max)
            bc_top = jnp.stack([x_t, y_t], axis=1)

            # Left: (0, y)
            x_l = jnp.full((N_bc,), x_min)
            y_l = jax.random.uniform(k3, (N_bc,), minval=y_min, maxval=y_max)
            bc_left = jnp.stack([x_l, y_l], axis=1)

            # Right: (1, y)
            x_r = jnp.full((N_bc,), x_max)
            y_r = jax.random.uniform(k4, (N_bc,), minval=y_min, maxval=y_max)
            bc_right = jnp.stack([x_r, y_r], axis=1)

            xy_bc_all = jnp.concatenate([bc_bottom, bc_top, bc_left, bc_right], axis=0)

        u_pred_bc = vmap(lambda xy: beta_apply_fn(params_beta, xy, theta))(xy_bc_all)
        
        u_target = 0.0
        
        return jnp.mean((u_pred_bc - u_target)**2)

    if rng_key is not None:
        bc_keys = jax.random.split(key_bc, B) # (B,) keys
        # vmap over (theta_batch, bc_keys)
        batch_bc_losses = vmap(bc_loss_single_system)(theta_batch, bc_keys)
    else:
        batch_bc_losses = vmap(lambda theta: bc_loss_single_system(theta, None))(theta_batch)

    loss_bc = jnp.mean(batch_bc_losses)
    
    lambda_bc = 100.0 
    
    return loss_pde + lambda_bc * loss_bc


def vmap_batch_mc_expectation(batched_parameters,H_mats,y_obser,params_beta,beta_apply_fn):
    
    return vmap(log_posterior,in_axes=(0,None,None,None,None))(batched_parameters,H_mats,y_obser,params_beta,beta_apply_fn).mean()


def alpha_loss_function(parameters,H_mats,y_obser,params_beta,beta_apply_fn):

    return - vmap_batch_mc_expectation(parameters,H_mats,y_obser,params_beta,beta_apply_fn)
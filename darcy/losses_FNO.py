import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
from fno import *
from utils import *
from data_generation import *
from constant_FNO_physics import *
from langevin_FNO import *


def create_model_a_fun(X, Y, z_coeffs, alpha_apply_fn):
    pi = jnp.pi
    phi_1 = jnp.sin(2 * pi * X) * jnp.sin(2 * pi * Y)
    phi_2 = jnp.sin(2 * pi * X) * jnp.sin(pi * Y)
    phi_3 = jnp.sin(pi * X) * jnp.sin(2 * pi * Y)
    basis_functions = jnp.stack([phi_1, phi_2, phi_3], axis=0)
    # basis_functions = jnp.stack([X * Y,X**2,Y**2],axis=0)

    def a_fun(params_alpha, u):
        Z_x_sum = jnp.einsum('j,jxy->xy', z_coeffs, basis_functions)
        Z_x_field = nn.softplus(Z_x_sum)
        # Z_x_field = Z_x_sum

        # Calculate g(u) using the MLP
        g_u = alpha_apply_fn({'params': params_alpha}, u.reshape(-1,1))

        # Combine everything using multiplication
        return Z_x_field * sigmoid_fn(g_u.reshape(nx, ny))
        # return Z_x_field * (jnp.exp(u) + g_u.reshape(nx, ny))
        # return Z_x_field * (1 + g_u.reshape(nx, ny))
        
    return a_fun


def physics_residual(u_pred, z_coeffs, params_alpha, alpha_apply_fn):

    a_fun = create_model_a_fun(X, Y, z_coeffs, alpha_apply_fn)
    a_full = a_fun(params_alpha, u_pred)

    f_weak = rhs_weakform(f_full, hx, hy)
    Au = matvec_operator(u_pred[1:-1,1:-1], a_full, nx, ny)
    residual = jnp.linalg.norm(Au - f_weak) / f_weak.size

    return residual


def batched_physics_residual(u_preds, batched_zs, params_alpha, alpha_apply_fn):

    return vmap(physics_residual, in_axes=(0, 0, None, None))(u_preds, batched_zs, params_alpha, alpha_apply_fn).sum()


def fno_physics_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn):
    inputs = jnp.tile(theta_batch[:, None, None, :], (1, nx-2, ny-2, 1))
    inputs = jnp.concatenate([inputs, grid_tiled[:10,1:-1,1:-1,:]], axis=-1)
    batched_zs = jnp.mean(inputs[:, :, :, 0:cfg.data_systems.no_parameters],axis=(1,2))
    out = beta_apply_fn(params_beta, inputs).squeeze(-1)
    
    pred = jnp.zeros((theta_batch.shape[0], nx, ny))
    pred = pred.at[:,1:-1,1:-1].set(out)

    physics_loss = batched_physics_residual(pred, batched_zs, params_alpha, alpha_apply_fn)
    return physics_loss / theta_batch.shape[0]
    # inputs = jnp.tile(theta_batch[:, None, None, :], (1, nx, ny, 1))
    # inputs = jnp.concatenate([inputs, grid_tiled], axis=-1)
    # batched_zs = jnp.mean(inputs[:, :, :, 0:cfg.data_systems.no_parameters],axis=(1,2))
    # out = beta_apply_fn(params_beta, inputs).squeeze(-1)

    # physics_loss = batched_physics_residual(out, batched_zs, params_alpha, alpha_apply_fn)
    # return physics_loss / theta_batch.shape[0]
    

def vmap_batch_mc_expectation(batched_parameters,H_mats,y_obser,params_beta,beta_apply_fn):
    
    return vmap(log_posterior,in_axes=(0,None,None,None,None))(batched_parameters,H_mats,y_obser,params_beta,beta_apply_fn).mean()


def alpha_loss_function(parameters,H_mats,y_obser,params_beta,beta_apply_fn):

    return - vmap_batch_mc_expectation(parameters,H_mats,y_obser,params_beta,beta_apply_fn)
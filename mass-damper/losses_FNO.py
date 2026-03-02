import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
from data_generation import *
from fno import *
from utils import *
from constant_FNO_physics import *
from langevin_FNO import *

def finite_difference(x, dt=cfg.data_systems.dt):
    dx = (x[:, 2:] - x[:, :-2]) / (2 * dt)
    return dx


@partial(jax.jit, static_argnames=('alpha_apply_fn', 'beta_apply_fn'))
def fno_physics_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn):
    # --- Forward simulations using the FNO model ---
    inputs = jnp.tile(theta_batch[:, None, :], (1, T-1, 1))
    t_broadcasted = jnp.tile(t[None, 1:, None], (inputs.shape[0], 1, 1)) 
    k, x0, v0 = jnp.exp(jnp.mean(inputs[:, :, 0:1],axis=1)), jnp.mean(inputs[:, :, 1:2],axis=1), jnp.mean(inputs[:, :, 2:3],axis=1)
    inputs = jnp.concatenate([inputs, t_broadcasted], axis=-1)
    
    out = beta_apply_fn(params_beta, inputs)
    initial_conditions = jnp.concatenate([x0, v0], axis=1)[:, None, :]
    out = jnp.concatenate([initial_conditions, out], axis=1)
    
    # --- Compute the physics loss ---
    # first residual 
    u_dot = finite_difference(out[:,:,0])
    residual = u_dot-out[:,1:-1,1]
    first_loss = jnp.mean(jnp.sum(residual ** 2, axis=1))
    
    # Second residual
    v_dot = finite_difference(out[:,:,1])
    velocity = out[:,1:-1,1][:,:,None]
    batched_apply = vmap(
        lambda x: alpha_apply_fn(params_alpha, x),
        in_axes=0 
    )
    output = batched_apply(velocity)[:,:,0]
    residual2 = v_dot - (forcing[1:-1].reshape(1,-1) - output - k * out[:,1:-1,0])
    second_loss = jnp.mean(jnp.sum(residual2 ** 2, axis=1))
    
    # Residual Gradient
    '''x_triple_dot = finite_difference(v_dot)
    
    # gradient of alpha w.r.t. x'
    def alpha_scalar_fn(x_single):
        return alpha_apply_fn(params_alpha, x_single[None, :])[0, 0]
    
    grad_alpha_fn = vmap(grad(alpha_scalar_fn))  # batched grad
    v_input = out[:, 1:-1, 1]  # x' aligned with v_dot
    v_flat = v_input.reshape(-1, 1)
    grad_alpha = grad_alpha_fn(v_flat).reshape(v_input.shape)  # shape: (batch, T-2)
    
    grad_term = grad_alpha * v_dot  # f_alpha'(x') * x''

    residual3 = x_triple_dot + grad_term[:, 1:-1] + k * u_dot[:,1:-1] - forcing_gradient[2:-2].reshape(1, -1)
    third_loss = jnp.mean(jnp.sum(residual3 ** 2, axis=1))'''

    return first_loss + second_loss


def fno_physics_loss_with_obser(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn):
    # --- Forward simulations using the FNO model ---
    inputs = jnp.tile(theta_batch[:, None, :], (1, T-1, 1))
    t_broadcasted = jnp.tile(t[None, 1:, None], (inputs.shape[0], 1, 1)) 
    k, x0, v0 = jnp.exp(jnp.mean(inputs[:, :, 0:1],axis=1)), jnp.mean(inputs[:, :, 1:2],axis=1), jnp.mean(inputs[:, :, 2:3],axis=1)
    inputs = jnp.concatenate([inputs, t_broadcasted], axis=-1)
    
    out = beta_apply_fn(params_beta, inputs)
    initial_conditions = jnp.concatenate([x0, v0], axis=1)[:, None, :]
    out = jnp.concatenate([initial_conditions, out], axis=1)
    
    # --- Compute the physics loss ---
    # first residual 
    u_dot = finite_difference(out[:,:,0])
    residual = u_dot-out[:,1:-1,1]
    first_loss = jnp.mean(jnp.sum(residual ** 2, axis=1))
    
    # Second residual
    v_dot = finite_difference(out[:,:,1])
    velocity = out[:,1:-1,1][:,:,None]
    batched_apply = vmap(
        lambda x: alpha_apply_fn(params_alpha, x),
        in_axes=0 
    )
    output = batched_apply(velocity)[:,:,0]
    residual2 = v_dot - (forcing[1:-1].reshape(1,-1) - output - k * out[:,1:-1,0])
    second_loss = jnp.mean(jnp.sum(residual2 ** 2, axis=1))
    
    # observation loss
    H = observation_matrix(20)
    # simulations, _ = vmap_batched_simulator_running(theta_batch,params_alpha, alpha_apply_fn)
    simulations, _ = vmap_batched_simulator_running(theta_batch,jax.lax.stop_gradient(params_alpha), alpha_apply_fn)
    observation_loss = jnp.linalg.norm(jnp.dot(simulations, H.T) - jnp.dot(out[:,:,0], H.T))**2

    return first_loss + second_loss + observation_loss


@partial(jax.jit, static_argnames=('alpha_apply_fn', 'beta_apply_fn'))
def supervised_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn):
    # --- Forward simulations using the FNO model ---
    inputs = jnp.tile(theta_batch[:, None, :], (1, T-1, 1))
    t_broadcasted = jnp.tile(t[None, 1:, None], (inputs.shape[0], 1, 1)) 
    k, x0, v0 = jnp.exp(jnp.mean(inputs[:, :, 0:1],axis=1)), jnp.mean(inputs[:, :, 1:2],axis=1), jnp.mean(inputs[:, :, 2:3],axis=1)
    inputs = jnp.concatenate([inputs, t_broadcasted], axis=-1)
    
    out = beta_apply_fn(params_beta, inputs)
    initial_conditions = jnp.concatenate([x0, v0], axis=1)[:, None, :]
    out = jnp.concatenate([initial_conditions, out], axis=1)
    
    simulations, velocity_paths = vmap_batched_simulator_running(theta_batch,params_alpha, alpha_apply_fn)
    observation_loss = jnp.linalg.norm(simulations - out[:,:,0])**2 + jnp.linalg.norm(velocity_paths - out[:,:,1])**2

    return observation_loss
    

def vmap_batch_mc_expectation(batched_parameters,y_obser,params_beta,beta_apply_fn):
    
    return vmap(log_posterior,in_axes=(0,None,None,None))(batched_parameters,y_obser,params_beta,beta_apply_fn).mean()


def alpha_loss_function(parameters,y_obser,params_beta,beta_apply_fn):
    
    return - vmap_batch_mc_expectation(parameters,y_obser,params_beta,beta_apply_fn)


@partial(jax.jit, static_argnames=('alpha_apply_fn',))
def G_leapfrog_running(parameters,params_alpha,alpha_apply_fn):
    
    def step_fn(carry, x):
        
        x1, x2, t = carry

        forcing = lambda t,x1,x2: (forcing_term(1,t) - alpha_apply_fn(params_alpha, jnp.array([x2])).reshape(x1.shape) - k*x1)/m

        x2 = x2 + 0.5 * dt * forcing(t,x1,x2)
        x1 = x1 + dt * x2
        x2 = x2 + 0.5 * dt * forcing(t+dt,x1,x2)
        
        t += dt
        
        return (x1,x2,t),x1

    m = 1.0
    k = jnp.exp(parameters[0])
    z0 = parameters[1]
    z0_dot = parameters[2]
    
    _, path = jax.lax.scan(step_fn, (z0,z0_dot,0.), xs=None, length=T-1)
    path = jnp.concatenate([jnp.array([z0]),path])
    
    return path


@partial(jax.jit, static_argnames=('alpha_apply_fn', ))
def G_rk4_running(parameters,params_alpha,alpha_apply_fn):
    
    def f(t, x1, x2):
        return (x2, (forcing_term(1, t) - alpha_apply_fn(params_alpha, jnp.array([x2])).reshape(x1.shape) - k * x1) / m)

    def step_fn(carry, _):
        x1, x2, t = carry

        k1_1, k1_2 = f(t, x1, x2)
        k2_1, k2_2 = f(t + dt/2, x1 + dt/2 * k1_1, x2 + dt/2 * k1_2)
        k3_1, k3_2 = f(t + dt/2, x1 + dt/2 * k2_1, x2 + dt/2 * k2_2)
        k4_1, k4_2 = f(t + dt, x1 + dt * k3_1, x2 + dt * k3_2)

        x1_new = x1 + (dt / 6) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        x2_new = x2 + (dt / 6) * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)

        t_new = t + dt

        return (x1_new, x2_new, t_new), (x1_new, x2_new)

    # Constants and parameters
    m = 1.0
    k = jnp.exp(parameters[0])
    z0 = parameters[1]
    z0_dot = parameters[2]

    # Run the integration
    _, (path, velocity_path) = jax.lax.scan(step_fn, (z0, z0_dot, 0.), xs=None, length=T-1)
    
    path = jnp.concatenate([jnp.array([z0]), path])
    velocity_path = jnp.concatenate([jnp.array([z0_dot]), velocity_path])

    return path, velocity_path


def vmap_batched_simulator_running(batched_parameters,params_alpha, alpha_apply_fn):
    return vmap(G_rk4_running,in_axes=(0,None,None))(batched_parameters,params_alpha, alpha_apply_fn)
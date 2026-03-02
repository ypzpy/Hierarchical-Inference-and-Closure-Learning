import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
from data_generation import *
from fno import *
from constant_PINNs_nonhierarchy import *
from langevin_PINNs_nonhierarchy import *
from utils_nonhierarchy import *


def finite_difference(x, dt=cfg.data_systems.dt):
    dx = (x[:, 2:] - x[:, :-2]) / (2 * dt)
    # dx = dx.at[:,1:-1].set((-x[:, 4:] + 8 * x[:, 3:-1] - 8 * x[:, 1:-3] + x[:, :-4]) / (12 * dt))
    return dx


def pinn_physics_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn, y_obser=None, rng_key=None):
    
    B = theta_batch.shape[0]
    T_points = 100
    if rng_key is not None:
        t_colloc = jax.random.uniform(rng_key, (B, T_points), minval=0.0, maxval=cfg.data_systems.stop_time)
    else:
        t_colloc = jnp.tile(jnp.linspace(0, cfg.data_systems.stop_time, T_points), (B, 1))

    def get_derivatives(t, theta):
        u_fn = lambda _t: beta_apply_fn(params_beta, _t, theta)
        
        u = u_fn(t)
        v = grad(u_fn)(t)
        a = grad(grad(u_fn))(t)
        return u, v, a

    def ode_residual_single_point(t, theta):
        k = jnp.exp(theta[0])
        u, v, a = get_derivatives(t, theta)
        
        f_pred = alpha_apply_fn(params_alpha, v.reshape(1,))[0]
        
        res = a + f_pred + k * u - 10.0 * jnp.sin(t)
        return res**2

    residual_over_time_fn = vmap(ode_residual_single_point, in_axes=(0, None))
    
    batch_residuals = vmap(residual_over_time_fn, in_axes=(0, 0))(t_colloc, theta_batch)
    
    loss_ode = jnp.mean(batch_residuals)
    
    # --- IC Loss ---
    def ic_error_single_system(theta):
        u0_true, v0_true = theta[1], theta[2]
        t0 = jnp.array(0.0)
        u_pred, v_pred, _ = get_derivatives(t0, theta)
        return (u_pred - u0_true)**2 + (v_pred - v0_true)**2

    batch_ic_errors = vmap(ic_error_single_system)(theta_batch)
    loss_ic = jnp.mean(batch_ic_errors)
    
    return loss_ode + loss_ic
    

def vmap_batch_mc_expectation(batched_parameters,y_obser,params_beta,beta_apply_fn):
    
    return vmap(log_posterior,in_axes=(0,None,None,None))(batched_parameters,y_obser,params_beta,beta_apply_fn).mean()


def alpha_loss_function(parameters,y_obser,params_beta,beta_apply_fn):
    
    return - vmap_batch_mc_expectation(parameters,y_obser,params_beta,beta_apply_fn)


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
import jax
import jax.numpy as jnp
from utils import *
from constant_FNO_physics import *

def forcing_term(omega,t):
    
    return 10 * jnp.sin(omega*t)


def G_leapfrog_true(parameters):
    
    def step_fn(carry, x):
        
        x1, x2, t = carry
        
        f = lambda t,x1,x2: (forcing_term(1,t) - a * x2 - b * x2 ** 3 - k*x1)/m
        
        x2 = x2 + 0.5 * dt * f(t,x1,x2)
        x1 = x1 + dt * x2
        x2 = x2 + 0.5 * dt * f(t+dt,x1,x2)
        
        t += dt
        
        return (x1,x2,t),(x1,x2)
    
    dt = cfg.data_systems.dt
    m = 1.0
    k = jnp.exp(parameters[0])
    z0 = parameters[1]
    z0_dot = parameters[2]
    a = cfg.true_parameters.true_a
    b = cfg.true_parameters.true_b

    _, (path,velocity_path) = jax.lax.scan(step_fn, (z0,z0_dot,0.), xs=None, length=T-1)
    path = jnp.concatenate([jnp.array([z0]),path])
    velocity_path = jnp.concatenate([jnp.array([z0_dot]),velocity_path])
    
    return path,velocity_path


def G_rk4_true(parameters):
    
    def f(t, x1, x2):
        return (x2, (forcing_term(1, t) - a * x2 - b * x2**3 - k * x1) / m)

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
    a = cfg.true_parameters.true_a
    b = cfg.true_parameters.true_b

    # Run the integration
    _, (path, velocity_path) = jax.lax.scan(step_fn, (z0, z0_dot, 0.), xs=None, length=T-1)
    
    path = jnp.concatenate([jnp.array([z0]), path])
    velocity_path = jnp.concatenate([jnp.array([z0_dot]), velocity_path])

    return path, velocity_path


@jit
def vmap_batched_simulator(batched_parameters):
    return vmap(G_rk4_true)(batched_parameters)


def observation_matrix(steps_per_observation):
    
    total_observation = int((T - 1)/steps_per_observation) + 1
    H = jnp.zeros((total_observation,T))
    
    for i in range(total_observation):
        H = H.at[i,i*steps_per_observation].set(1)
        
    return H


def obtain_observations(parameter_matrix, rng_key):
    key, rng_key = jax.random.split(rng_key,2)
    simulations, _ = vmap_batched_simulator(parameter_matrix)
    
    if jnp.isnan(simulations).any():
        print("Diverging simulations detected")
        
    noisy_data = simulations + cfg.data_systems.obser_noise * jax.random.normal(key, shape=simulations.shape)
    
    # take sparse observations
    H = observation_matrix(cfg.data_systems.obser_freq)
    y = jnp.dot(noisy_data,H.T)
    
    return y
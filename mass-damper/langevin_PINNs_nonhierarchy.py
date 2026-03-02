import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
from data_generation import *
from fno import *
from constant_PINNs_nonhierarchy import *
from utils_nonhierarchy import *


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: 
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    keys_tree = random_split_like_tree(rng_key, target)
    return tree_map(lambda l, k: mean + std * jax.random.normal(k, shape=l.shape, dtype=l.dtype), target, keys_tree)


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def log_posterior_for_systems(batched_parameters,batched_data,tau_phi,mu_phi,params_beta,beta_apply_fn):

    t_eval = t

    def predict_over_time(time_array, p_vec):
        return jax.vmap(lambda _t: beta_apply_fn(params_beta, _t, p_vec))(time_array)

    predict_batch = jax.vmap(predict_over_time, in_axes=(None, 0))

    out = predict_batch(t_eval, batched_parameters)
    out = out[..., None]

    simulation = out[:,:,0]
    
    H = observation_matrix(cfg.data_systems.obser_freq)
    log_likelihood = stats.norm.logpdf(batched_data, jnp.dot(simulation, H.T), cfg.data_systems.obser_noise)
    log_prior = stats.norm.logpdf(batched_parameters,mu_phi,jnp.sqrt(tau_phi))
    
    return log_likelihood.sum() + log_prior.sum()


def log_posterior(parameters,y_obser,params_beta,beta_apply_fn):
    
    parameters_matrix = parameters[:].reshape(cfg.data_systems.n_systems,-1)
    mu_phi = jnp.array([jnp.log(cfg.prior.mean.k_mean),
                        cfg.prior.mean.z0_mean,
                        cfg.prior.mean.zdot_mean])
    tau_phi = jnp.array([cfg.prior.variance.k_variance,
                         cfg.prior.variance.z0_variance,
                         cfg.prior.variance.zdot_variance])

    log_theta_posterior = log_posterior_for_systems(
                                parameters_matrix,y_obser,tau_phi,mu_phi,params_beta,beta_apply_fn)

    return log_theta_posterior


def log_proposal(theta_star,theta, C, R, e, y_obser,params_beta,beta_apply_fn):

    grad = jax.grad(log_posterior)(theta, y_obser, params_beta, beta_apply_fn)
    diff = theta_star - theta - 0.5 * e ** 2 * C @ grad
    return -0.5 * (diff.T / e**2 @ jax.scipy.linalg.cho_solve((R.T,False),diff))
    
    
def single_MALA(parameters,e,rng_key,y_obser,C,R,params_beta,beta_apply_fn):
    
    key, uniform_key = jax.random.split(rng_key, 2)
        
    # generate random term
    W = normal_like_tree(key,parameters)
    
    # compute gradient of log posterior
    grad = jax.grad(log_posterior)(parameters, y_obser, params_beta, beta_apply_fn)

    # new_parameters = parameters + 0.5 * e ** 2 * C @ grad + 0.5 * e ** 2 * (parameters.shape[0] + 1) / args.num_chains * (parameters - parameters.mean()) + e * R @ W
    # parameters = new_parameters
    new_parameters = parameters + 0.5 * e ** 2 * C @ grad + e * R @ W
    
    # metropolis step
    log_accept_prob = log_posterior(new_parameters,y_obser,params_beta,beta_apply_fn) - log_posterior(parameters,y_obser,params_beta,beta_apply_fn) + \
                      log_proposal(parameters,new_parameters,C,R,e,y_obser,params_beta,beta_apply_fn) - log_proposal(new_parameters,parameters,C,R,e,y_obser,params_beta,beta_apply_fn)

    accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
    accept = (jax.random.uniform(uniform_key) < accept_prob)
    parameters = ifelse(accept, new_parameters, parameters)
    
    # adapt step size
    e = e * jnp.sqrt(1 + cfg.langevin_sampler.step_size_lr * (accept_prob - cfg.langevin_sampler.accept_ratio))
    
    return parameters, e


def ensemble_MALA(batched_parameters,batched_e,rng_key,C,R,y_obser,params_beta,beta_apply_fn):

    keys = jax.random.split(rng_key, cfg.langevin_sampler.n_chains)
    # keys = jax.random.split(rng_key, batch_size)

    return vmap(single_MALA,in_axes=(0,0,0,None,None,None,None,None))(batched_parameters,batched_e,keys,y_obser,C,R,params_beta,beta_apply_fn)


def inference_loop(rng_key,batched_parameters,batched_e,params_beta,step,C,R,y_obser,beta_apply_fn):

    for _ in range(cfg.langevin_sampler.n_samples):

        # for i in range(cfg.langevin_sampler.batch_num):
        #     key, rng_key = jax.random.split(rng_key,2)

        #     batched_parameters_batch, batched_e_batch = step(batched_parameters[i*batch_size:(i+1)*batch_size,:],batched_e[i*batch_size:(i+1)*batch_size,:],key,C,R,y_obser,params_beta,beta_apply_fn)
        #     batched_parameters = batched_parameters.at[i*batch_size:(i+1)*batch_size,:].set(batched_parameters_batch)
        #     batched_e = batched_e.at[i*batch_size:(i+1)*batch_size,:].set(batched_e_batch)
        
        key, rng_key = jax.random.split(rng_key,2)
        batched_parameters, batched_e = step(batched_parameters,batched_e,key,C,R,y_obser,params_beta,beta_apply_fn)
            
        C = jnp.cov(batched_parameters.T)
        R = jnp.linalg.cholesky(C + 0.000001 * jnp.eye(3 * (cfg.data_systems.n_systems)))
        # R = jnp.linalg.cholesky(C)
    
    return batched_parameters, batched_e, C, R   
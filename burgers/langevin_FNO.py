import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
from fno import *
from utils import *
from constant_FNO_supervised import *


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


def log_posterior_for_systems(parameter_matrix,H_mats,y_obser,masks,log_tau_phi,mu_phi,params_beta,beta_apply_fn):

    inputs = jnp.tile(parameter_matrix[:, None, None, :], (1, nt, nx, 1))
    inputs = jnp.concatenate([inputs, grid_tiled], axis=-1)
    
    out = beta_apply_fn(params_beta, inputs).squeeze(-1)

    # 4. Project to Observation Space
    # Flatten the full field: (batch, nt * nx)
    u_flat = out.reshape(parameter_matrix.shape[0], -1)
    
    # Matrix multiplication: y_pred = H @ u_flat
    # H_mats: (batch, n_obs, nt*nx), u_flat: (batch, nt*nx) -> y_pred: (batch, n_obs)
    # Using matmul with broadcasting logic or einsum (here keeping similar to your original style)
    y_pred = jnp.einsum('bij, bj -> bi', H_mats, u_flat)
    
    # 3. Calculate Likelihood with Masking (CRITICAL!)
    all_log_lik = jax.scipy.stats.norm.logpdf(
        y_obser, 
        y_pred, 
        cfg.data_systems.obser_noise
    )
    
    masked_log_lik = (all_log_lik * masks).sum()
    
    log_prior = stats.norm.logpdf(parameter_matrix,mu_phi,jnp.sqrt(jnp.exp(log_tau_phi)))
    
    return masked_log_lik + log_prior.sum()


def log_posterior(parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn):
    
    mu_phi = parameters[-2*cfg.data_systems.no_parameters:-cfg.data_systems.no_parameters]
    log_tau_phi = parameters[-cfg.data_systems.no_parameters:]
    parameter_matrix = parameters[:-2*cfg.data_systems.no_parameters].reshape(cfg.data_systems.n_systems,-1)

    # hyperprior mean
    mu01 = jnp.log(cfg.hyperprior.mean.hyperprior_z1_prior)
    mu02 = cfg.hyperprior.mean.hyperprior_z2_prior
    hyperprior_mean = jnp.array([mu01,mu02])
    # hyperprior_mean = jnp.array([mu01,mu02])

    hyper_mean_tau_z = jnp.array([cfg.hyperprior.mean.hyperprior_mean_tau_z1,cfg.hyperprior.mean.hyperprior_mean_tau_z2])

    # hyperprior variance
    hyper_variance_mean = jnp.log(jnp.array([cfg.hyperprior.variance.hyperprior_variance_mu_z1,
                                             cfg.hyperprior.variance.hyperprior_variance_mu_z2]))
    hyper_variance_tau = jnp.array([cfg.hyperprior.variance.hyperprior_variance_tau_z1,
                                    cfg.hyperprior.variance.hyperprior_variance_tau_z2])
    # hyper_variance_mean = jnp.log(jnp.array([cfg.hyperprior.variance.hyperprior_variance_mu_z1,
    #                                          cfg.hyperprior.variance.hyperprior_variance_mu_z2]))
    # hyper_variance_tau = jnp.array([cfg.hyperprior.variance.hyperprior_variance_tau_z1,
    #                                 cfg.hyperprior.variance.hyperprior_variance_tau_z2])
    
    # log p(phi)
    log_phi_prior_z = stats.norm.logpdf(mu_phi[:],hyperprior_mean[:],jnp.sqrt(hyper_mean_tau_z))
    log_tau_prior_z = stats.norm.logpdf(log_tau_phi[:],hyper_variance_mean[:],jnp.sqrt(hyper_variance_tau[:]))
    
    # terms for multiplication
    log_theta_posterior = log_posterior_for_systems(
                                parameter_matrix,H_mats,y_obser,masks,log_tau_phi,mu_phi,params_beta,beta_apply_fn)

    return log_phi_prior_z.sum() + log_tau_prior_z.sum() + log_theta_posterior


def log_proposal(theta_star,theta, H_mats, C, R, e, y_obser,masks,params_beta,beta_apply_fn):

    grad = jax.grad(log_posterior)(theta, H_mats, y_obser, masks, params_beta, beta_apply_fn)
    diff = theta_star - theta - 0.5 * e ** 2 * C @ grad
    return -0.5 * (diff.T / e**2 @ jax.scipy.linalg.cho_solve((R.T,False),diff))
    
    
def single_MALA(parameters,e,rng_key,H_mats,y_obser,masks,C,R,params_beta,beta_apply_fn):
    
    key, uniform_key = jax.random.split(rng_key, 2)
        
    # generate random term
    W = normal_like_tree(key,parameters)
    
    # compute gradient of log posterior
    grad = jax.grad(log_posterior)(parameters, H_mats, y_obser, masks, params_beta, beta_apply_fn)

    # new_parameters = parameters + 0.5 * e ** 2 * C @ grad + 0.5 * e ** 2 * (parameters.shape[0] + 1) / args.num_chains * (parameters - parameters.mean()) + e * R @ W
    # parameters = new_parameters
    new_parameters = parameters + 0.5 * e ** 2 * C @ grad + e * R @ W
    
    # metropolis step
    log_accept_prob = log_posterior(new_parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn) - log_posterior(parameters,H_mats,y_obser,masks,params_beta,beta_apply_fn) + \
                      log_proposal(parameters,new_parameters,H_mats,C,R,e,y_obser,masks,params_beta,beta_apply_fn) - log_proposal(new_parameters,parameters,H_mats,C,R,e,y_obser,masks,params_beta,beta_apply_fn)

    accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
    accept = (jax.random.uniform(uniform_key) < accept_prob)
    parameters = ifelse(accept, new_parameters, parameters)
    
    # adapt step size
    e = e * jnp.sqrt(1 + cfg.langevin_sampler.step_size_lr * (accept_prob - cfg.langevin_sampler.accept_ratio))
    
    return parameters, e


def ensemble_MALA(batched_parameters,batched_e,rng_key,H_mats,y_obser,masks,C,R,params_beta,beta_apply_fn):

    # keys = jax.random.split(rng_key, cfg.langevin_sampler.n_chains)
    keys = jax.random.split(rng_key, batch_size)

    return vmap(single_MALA,in_axes=(0,0,0,None,None,None,None,None,None,None))(batched_parameters,batched_e,keys,H_mats,y_obser,masks,C,R,params_beta,beta_apply_fn)


def inference_loop(rng_key,batched_parameters,batched_e,H_mats,params_beta,step,C,R,y_obser,masks,beta_apply_fn):

    for _ in range(cfg.langevin_sampler.n_samples):

        for i in range(cfg.langevin_sampler.batch_num):
            key, rng_key = jax.random.split(rng_key,2)

            batched_parameters_batch, batched_e_batch = step(batched_parameters[i*batch_size:(i+1)*batch_size,:],batched_e[i*batch_size:(i+1)*batch_size,:],key,H_mats,y_obser,masks,C,R,params_beta,beta_apply_fn)
            batched_parameters = batched_parameters.at[i*batch_size:(i+1)*batch_size,:].set(batched_parameters_batch)
            batched_e = batched_e.at[i*batch_size:(i+1)*batch_size,:].set(batched_e_batch)
        
        # key, rng_key = jax.random.split(rng_key,2)
        # batched_parameters, batched_e = step(batched_parameters,batched_e,key,C,R,y_obser,params_beta,beta_apply_fn)
            
        C = jnp.cov(batched_parameters.T)
        R = jnp.linalg.cholesky(C + 0.00001 * jnp.eye(2 * (cfg.data_systems.n_systems+2)))
        # R = jnp.linalg.cholesky(C)
    
    return batched_parameters, batched_e, C, R   
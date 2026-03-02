import numpy as np
import operator as op
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap, grad, value_and_grad, lax
from jax.tree_util import tree_map
import time
from functools import partial
import matplotlib.pyplot as plt
import jax.scipy.stats as stats
import os
import optax
from flax import linen as nn
from functools import partial
from datetime import date
from jax.scipy.sparse.linalg import cg as jax_cg
from jaxopt import FixedPointIteration
from flax.training import checkpoints
import wandb
from data_generation import *
from mlp import *
from utils import *
from constant_solver import *


def matvec_operator(v_interior, a_full, nx, ny):
    """
    Computes the matrix-vector product A @ v without forming A explicitly.
    """
    # Pad the interior vector `v` with zeros to form a full grid
    v_full = jnp.zeros((nx, ny)).at[1:-1, 1:-1].set(v_interior.reshape(nx-2, ny-2))

    # Dirichlet boundary => test functions only interior
    r = jnp.zeros((nx-2, ny-2))

    # ----- x-direction flux contributions -----
    r_x = jnp.zeros_like(r)
    # flux from west face
    r_x = r_x.at[:, :].add( (a_full[1:-1, :-2] + a_full[1:-1, 1:-1]) * (v_full[1:-1, 1:-1] - v_full[1:-1, :-2]) )
    # flux from east face
    r_x = r_x.at[:, :].add(-(a_full[1:-1, 1:-1] + a_full[1:-1, 2:]) * (v_full[1:-1, 2:] - v_full[1:-1, 1:-1]) )

    # ----- y-direction flux contributions -----
    r_y = jnp.zeros_like(r)
    # flux from south face
    r_y = r_y.at[:, :].add( (a_full[:-2, 1:-1] + a_full[1:-1, 1:-1]) * (v_full[1:-1, 1:-1] - v_full[:-2,1:-1]))
    # flux from north face
    r_y = r_y.at[:, :].add(-(a_full[2:, 1:-1] + a_full[1:-1, 1:-1]) * (v_full[2:, 1:-1] - v_full[1:-1, 1:-1]) )

    Av = 0.5 * (r_x + r_y)

    return Av.ravel()


def rhs_weakform(f_full, hx, hy):
    """
    Assemble RHS vector in weak form with hat basis.
    f_full: (nx, ny) values of source term
    returns flattened (nx-2)*(ny-2) vector
    """
    # interior quadrature: weighted average around node
    rhs = (hx*hy / 9.0) * (
        3 * f_full[1:-1, 1:-1] +
        f_full[:-2, :-2] + f_full[2:, 2:] +
        f_full[:-2, 1:-1] + f_full[2:, 1:-1] +
        f_full[1:-1, :-2] + f_full[1:-1, 2:]
    )
    return rhs.ravel()


def sigmoid_fn(x):
    return 1 / (1 + jnp.exp(-x))


def make_fixed_point_iteration_train(alpha_apply_fn, basis_functions, nx, ny, hx, hy):
    def fixed_point_iteration_train(u_full, parameters, mlp_params, static_params):
        z_coeffs = parameters

        f_full = static_params["f_full"]
        damping = static_params["damping"]

        Z_x_sum = jnp.einsum('j,jxy->xy', z_coeffs, basis_functions)
        Z_x_field = nn.softplus(Z_x_sum)

        g_u = alpha_apply_fn({'params': mlp_params}, u_full.reshape(-1,1))
        a_full = Z_x_field * sigmoid_fn(g_u.reshape(nx, ny))

        f_weak = rhs_weakform(f_full, hx, hy)

        matvec_fun = lambda v: matvec_operator(v, a_full, nx, ny)
        u_interior, _ = jax_cg(matvec_fun, f_weak, tol=1e-6)
        u_temp = u_full.at[1:-1, 1:-1].set(u_interior.reshape(nx-2, ny-2))

        u_new = (1.0 - damping) * u_full + damping * u_temp
        return u_new

    return fixed_point_iteration_train

    
def G_poisson_train(parameters, u_init, mlp_params, alpha_apply_fn):
    """
    Solve nonlinear Poisson equation for one system using learned MLP a(u).
    
    Args:
        parameters: (3,) forcing term coefficients
        mlp_params: parameters of MLP model (dict from Flax)
    
    Returns:
        u_sol: (nx, ny) solution field
    """
    f_full = f_source_term(X, Y)  # (nx, ny)

    static_params = {
        "f_full": f_full,
        "damping": 0.6,
    }
    # --- solve PDE via fixed-point iteration ---
    u_sol, _ = solver_train.run(u_init, parameters, mlp_params, static_params)
    
    return u_sol
    

@jit
def vmap_batched_poisson_train(batched_parameters, u_inits, mlp_params, alpha_apply_fn):
    """
    batched_parameters: (batch, 3)
    returns: (batch, nx, ny) PDE solutions
    """
    return vmap(G_poisson_train,in_axes=(0,0,None,None))(batched_parameters, u_inits, mlp_params, alpha_apply_fn)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: 
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    keys_tree = random_split_like_tree(rng_key, target)
    return tree_map(lambda l, k: mean + std*jax.random.normal(k, l.shape, l.dtype), target, keys_tree)


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def log_losterior_per_system(parameters,y_obser,H,u_init,log_tau_phi,sigma_obs,mu_phi,mlp_params,alpha_apply_fn):
    simulation = G_poisson_train(parameters,u_init,mlp_params,alpha_apply_fn)
    log_likelihood = stats.norm.logpdf(y_obser,jnp.dot(H,simulation.reshape(-1,1)).ravel(),sigma_obs)
    log_prior = stats.norm.logpdf(parameters,mu_phi,jnp.sqrt(jnp.exp(log_tau_phi)))
    
    return log_likelihood.sum() + log_prior.sum()


@partial(jax.jit, static_argnums=(8,))
def vmap_batched_log_losterior_per_system(batched_parameters,batched_data,batched_H,u_inits,log_tau_phi,sigma_obs,mu_phi,mlp_params,alpha_apply_fn):
    return vmap(log_losterior_per_system,in_axes=(0,0,0,0,None,None,None,None,None))(
                    batched_parameters,batched_data,batched_H,u_inits,log_tau_phi,sigma_obs,mu_phi,mlp_params,alpha_apply_fn)


def log_posterior(parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn):

    mu_phi = parameters[-6:-3]
    log_tau_phi = parameters[-3:]
    parameters_matrix = parameters[:-6].reshape(cfg.data_systems.n_systems,-1)
    sigma_obs = cfg.data_systems.obser_noise
    
    # hyperprior mean
    mu01 = cfg.hyperprior.mean.hyperprior_z1_prior
    mu02 = cfg.hyperprior.mean.hyperprior_z2_prior
    mu03 = cfg.hyperprior.mean.hyperprior_z3_prior
    hyperprior_mean = jnp.array([mu01,mu02,mu03])

    hyper_mean_tau_z = cfg.hyperprior.mean.hyperprior_mean_tau_z

    # hyperprior variance
    hyper_variance_mean = jnp.log(jnp.array([cfg.hyperprior.variance.hyperprior_variance_mu_z1,
                                             cfg.hyperprior.variance.hyperprior_variance_mu_z2,
                                             cfg.hyperprior.variance.hyperprior_variance_mu_z3]))
    hyper_variance_tau = jnp.array([cfg.hyperprior.variance.hyperprior_variance_tau_z1,
                                    cfg.hyperprior.variance.hyperprior_variance_tau_z2,
                                    cfg.hyperprior.variance.hyperprior_variance_tau_z3])
    
    # log p(phi)
    log_phi_prior_z = stats.norm.logpdf(mu_phi[:],hyperprior_mean[:],jnp.sqrt(hyper_mean_tau_z))
    log_tau_prior_z = stats.norm.logpdf(log_tau_phi[:],hyper_variance_mean[:],jnp.sqrt(hyper_variance_tau[:]))
    
    # terms for multiplication
    log_theta_posterior = vmap_batched_log_losterior_per_system(
                                parameters_matrix,y_obser,H_mats,u_inits,log_tau_phi,sigma_obs,mu_phi,mlp_params,alpha_apply_fn)

    return log_phi_prior_z.sum() + log_tau_prior_z.sum() + log_theta_posterior.sum()


def log_proposal(theta_star,theta, y_obser, H_mats, u_inits, C, R, e, mlp_params, alpha_apply_fn):

    grad = jax.grad(log_posterior)(theta,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn)
    diff = theta_star - theta - 0.5 * e ** 2 * C @ grad
    return -0.5 * (diff.T / e**2 @ jax.scipy.linalg.cho_solve((R.T,False),diff))
    
    
@partial(jax.jit, static_argnums=(9,))
def single_MALA(parameters,e,rng_key,y_obser,H_mats,u_inits,C,R,mlp_params,alpha_apply_fn):
    
    key, uniform_key = jax.random.split(rng_key, 2)
        
    # generate random term
    W = normal_like_tree(key,parameters)
    
    # compute gradient of log posterior
    grad = jax.grad(log_posterior)(parameters, y_obser, H_mats, u_inits, mlp_params, alpha_apply_fn)
        
    new_parameters = parameters + 0.5 * e ** 2 * C @ grad + e * R @ W
    
    # metropolis step
    log_accept_prob = log_posterior(new_parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn) - log_posterior(parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn) + \
                      log_proposal(parameters,new_parameters,y_obser,H_mats,u_inits,C,R,e,mlp_params,alpha_apply_fn) - log_proposal(new_parameters,parameters,y_obser,H_mats,u_inits,C,R,e,mlp_params,alpha_apply_fn)

    accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
    accept = (jax.random.uniform(uniform_key) < accept_prob)
    parameters = ifelse(accept, new_parameters, parameters)
    
    # adapt step size
    e = e * jnp.sqrt(1 + cfg.langevin_sampler.step_size_lr * (accept_prob - cfg.langevin_sampler.accept_ratio))
    
    return parameters, e


def ensemble_MALA(batched_parameters,batched_e,rng_key,y_obser,H_mats,u_inits,C,R,mlp_params,alpha_apply_fn):
    
    # keys = jax.random.split(rng_key, args.num_chains)
    
    keys = jax.random.split(rng_key, batch_size)

    return vmap(single_MALA,in_axes=(0,0,0,None,None,None,None,None,None,None))(batched_parameters,batched_e,keys,y_obser,H_mats,u_inits,C,R,mlp_params,alpha_apply_fn)


def inference_loop(rng_key,batched_parameters,batched_e,y_obser,H_mats,u_inits,mlp_params,step,C,R,alpha_apply_fn):
    
    for _ in range(cfg.langevin_sampler.n_samples):

        for i in range(cfg.langevin_sampler.batch_num):
            key, rng_key = jax.random.split(rng_key,2)

            batched_parameters_batch, batched_e_batch = step(batched_parameters[i*batch_size:(i+1)*batch_size,:],batched_e[i*batch_size:(i+1)*batch_size,:],key,y_obser,H_mats,u_inits,C,R,mlp_params,alpha_apply_fn)
            batched_parameters = batched_parameters.at[i*batch_size:(i+1)*batch_size,:].set(batched_parameters_batch)
            batched_e = batched_e.at[i*batch_size:(i+1)*batch_size,:].set(batched_e_batch)
        
        # key, rng_key = jax.random.split(rng_key,2)
        # batched_parameters, batched_e = step(batched_parameters,batched_e,key,C,R,y_obser,params_beta,beta_apply_fn)
            
        C = jnp.cov(batched_parameters.T)
        # R = jnp.linalg.cholesky(C + 0.00001 * jnp.eye(3 * (args.num_of_systems+2)))
        R = jnp.linalg.cholesky(C)
    
    return batched_parameters, batched_e, C, R


def vmap_batch_mc_expectation(batched_parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn):
    
    return vmap(log_posterior,in_axes=(0,None,None,None,None,None))(batched_parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn).mean()


def loss_function(parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn):
    
    return - vmap_batch_mc_expectation(parameters,y_obser,H_mats,u_inits,mlp_params,alpha_apply_fn)


def l2_error(params,alpha_apply_fn,max_value):
    
    x = jnp.linspace(0.0,max_value,40).reshape(-1,1)
    
    y_true = x ** 2
    y_pred = alpha_apply_fn({'params': params},x)
    
    return jnp.linalg.norm(y_pred - y_true, ord=2)
    
    
solver_train = None


def main(cfg):
    
    rng_key = jax.random.key(cfg.data_systems.rng_key)
    
    print("Initialising wandb...")
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=vars(cfg),
        group=cfg.wandb.group,
    )
    checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Checkpoint directory:", checkpoint_dir)
    print("Done.")

    print("Generating GT parameter matrix and observation data...")
    key, rng_key = jax.random.split(rng_key,2)
    parameter_matrix = generate_parameter_set(cfg, key)
    systems = vmap_batched_poisson_jaxopt(parameter_matrix)
    save_path = os.path.join(checkpoint_dir, "parameters.npy")
    jnp.save(save_path,parameter_matrix)
    
    key, rng_key = jax.random.split(rng_key,2)
    H_mats, idxs, y_obser = obtain_observations(parameter_matrix, key)
    u_inits = jnp.zeros((cfg.data_systems.n_systems, nx, ny))
    print("     Maximum u value: ", systems.max())
    print("Done.")
    
    print("Initialising batched parameters and langevin chain...")

    key, rng_key = jax.random.split(rng_key,2)
    batched_parameters = vmap_single_chain_initialisation(key).reshape(cfg.langevin_sampler.n_chains,-1)
    C = jnp.cov(batched_parameters.T)
    R = jnp.linalg.cholesky(C)
    batched_e = jnp.tile(jnp.array([cfg.langevin_sampler.step_size]),(cfg.langevin_sampler.n_chains,1))
    step = jax.jit(ensemble_MALA,static_argnames=['alpha_apply_fn'])
    whole_chain = jnp.empty((3, 0, batched_parameters.shape[-1]))
    print("Done.")
    
    print("Initialising alpha model...")
    alpha_key, rng_key = random.split(rng_key, 2)

    alpha_model = MLP(cfg)
    params_alpha = alpha_model.init(alpha_key, jnp.zeros((1,1)))['params']
    lr = optax.exponential_decay(
        cfg.models.mlp_alpha.learning_rate,
        cfg.models.mlp_alpha.n_decay_steps,
        cfg.models.mlp_alpha.decay_rate,
        staircase=True
    )
    # alpha_optimiser = optax.adamw(learning_rate=lr, weight_decay=cfg.models.mlp_alpha.weight_decay)
    alpha_optimiser = optax.adam(lr)
    opt_state_alpha = alpha_optimiser.init(params_alpha)
    alpha_apply_fn = partial(alpha_model.apply)
    print("Done.")
    
    pi = jnp.pi
    phi_1 = jnp.sin(pi * X) * jnp.sin(pi * Y)
    phi_2 = jnp.sin(2 * pi * X) * jnp.sin(pi * Y)
    phi_3 = jnp.sin(pi * X) * jnp.sin(2 * pi * Y)
    basis_functions = jnp.stack([phi_1, phi_2, phi_3], axis=0)

    global solver_train
    fixed_point_iteration_train = make_fixed_point_iteration_train(alpha_apply_fn, basis_functions, nx, ny, hx, hy)
    jitted_fixed_point_train = jax.jit(fixed_point_iteration_train)
    solver_train = FixedPointIteration(
        fixed_point_fun=jitted_fixed_point_train,
        maxiter=10,
        tol=1e-5,
        jit=True,
        implicit_diff=True
    )
    
    print("Starting training...")
    best_l2_error = float('inf')
    
    for i in range(10000):
        start_time = time.time()
        
        key, rng_key = jax.random.split(rng_key,2)
        
        batched_parameters, batched_e, C, R = inference_loop(key,batched_parameters,batched_e,y_obser,H_mats,u_inits,params_alpha,step,C,R,alpha_apply_fn)
        whole_chain = jnp.concatenate([whole_chain, batched_parameters[0:3, None, :]], axis=1)
        
        grads_accum = jax.tree_util.tree_map(jnp.zeros_like, params_alpha)
        loss_accum = 0.0

        for j in range(0, cfg.langevin_sampler.n_chains, batch_size):
            batch = batched_parameters[j:j+batch_size]

            (val,grad) = jax.value_and_grad(loss_function, argnums=4)(batch, y_obser,H_mats, u_inits, params_alpha,alpha_apply_fn)

            grads_accum = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, grads_accum, grad)
            loss_accum += val

        num_batches = cfg.langevin_sampler.batch_num
        grads_accum = jax.tree_util.tree_map(lambda g: g / num_batches, grads_accum)

        updates, opt_state_alpha = alpha_optimiser.update(grads_accum, opt_state_alpha, params_alpha)
        params_alpha = optax.apply_updates(params_alpha, updates)

        error = l2_error(params_alpha, alpha_apply_fn, systems.max())

        log_data = {
                "Loss/Alpha Function Loss": error,
                "Loss/MCMC Loss": loss_accum,
                "Hyperprior/mu_z1": batched_parameters[0][-6],
                "Hyperprior/mu_z2": batched_parameters[0][-5],
                "Hyperprior/mu_z3": batched_parameters[0][-4],
                "Hyperprior/tau_z1": jnp.exp(batched_parameters[0][-3]),
                "Hyperprior/tau_z2": jnp.exp(batched_parameters[0][-2]),
                "Hyperprior/tau_z3": jnp.exp(batched_parameters[0][-1]),
            }
        
        if i % 50 == 0 and i > 0:
            fig_force, ax_force = plt.subplots(figsize=(8, 6))
            x_plot = jnp.linspace(0.0, systems.max(), 50).reshape(-1, 1)
            y_true_plot = x_plot ** 2
            y_pred_plot = alpha_model.apply({'params': params_alpha},x_plot)
            ax_force.plot(x_plot, y_true_plot, label='true')
            ax_force.plot(x_plot, y_pred_plot, label='predict', linestyle='--')
            ax_force.legend()
            ax_force.set_title(f'Alpha Function Fit @ Epoch {i}')
            ax_force.grid(True)
            log_data["Plots/Alpha Function Fit"] = wandb.Image(fig_force)
            plt.close(fig_force)
                
        wandb.log(log_data, step=i)
        
        if error < best_l2_error:
            best_l2_error = error
            wandb.summary['best_l2_error'] = best_l2_error
            
            best_params = {'alpha': params_alpha}
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=best_params,
                step=1, 
                prefix='best_model_',
                overwrite=True 
            )
            print(f"Epoch {i}: New best model saved with L2 Error: {best_l2_error:.6f}")
            
        if i % 200 == 0 and i > 0:
            current_params = {'alpha': params_alpha}
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=current_params,
                step=1, 
                prefix='current_model_',
                overwrite=True 
            )
            print(f"Epoch {i}: New current model saved with L2 Error: {error:.6f}")

        if i % 200 == 0 and i > 0:
            save_path = os.path.join(checkpoint_dir, "whole_chain.npy")
            np.save(save_path, np.array(whole_chain))
            print(f"Saved whole_chain snapshot at epoch {i} -> {save_path}")

        end_time = time.time()
        print(f"Epoch {i} completed in {end_time - start_time:.2f} seconds.")

        print(f"L2 Error, alpha updating iterations {i}, error: {error}, loss:", loss_accum)
        mean_of_chain = batched_parameters.mean(axis=0)
        print("      z1, z2, z3 = ", mean_of_chain[-6:-3])
        print("      variance = ", jnp.exp(mean_of_chain)[-3:])
            

if __name__ == "__main__":
    
    cfg = load_config()
    main(cfg)






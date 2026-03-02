import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from functools import partial
import matplotlib.pyplot as plt
from jax.tree_util import tree_map
import optax
from functools import partial
import flax
from datetime import date
import pickle
import wandb
import time
from flax.training import checkpoints
from data_generation import *
from fno import *
from utils import *
from constant_solver import *
from mlp import *
from jax import checkpoint as remat


def observation_matrix(steps_per_observation):
    
    total_observation = int((T - 1)/steps_per_observation) + 1
    H = jnp.zeros((total_observation,T))
    
    for i in range(total_observation):
        H = H.at[i,i*steps_per_observation].set(1)
        
    return H


@partial(jax.jit, static_argnames=('alpha_apply_fn',))
def G_leapfrog(parameters,params_alpha,alpha_apply_fn):
        
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


def log_losterior_per_system(parameters,data,log_tau_phi,sigma_obs,mu_phi,params_alpha, alpha_apply_fn):
    simulation = G_leapfrog(parameters,params_alpha,alpha_apply_fn)
    H = observation_matrix(cfg.data_systems.obser_freq)
    log_likelihood = stats.norm.logpdf(data,jnp.dot(H,simulation),sigma_obs)
    log_prior = stats.norm.logpdf(parameters,mu_phi,jnp.sqrt(jnp.exp(log_tau_phi)))
    
    return log_likelihood.sum() + log_prior.sum()


def vmap_batched_log_losterior_per_system(batched_parameters,batched_data,log_tau_phi,sigma_obs,mu_phi,params_alpha, alpha_apply_fn):
    return vmap(log_losterior_per_system,in_axes=(0,0,None,None,None,None,None))(
                    batched_parameters,batched_data,log_tau_phi,sigma_obs,mu_phi,params_alpha, alpha_apply_fn).sum()
    
    
def log_posterior(parameters,params_alpha,alpha_apply_fn,y_obser):
    
    mu_phi = parameters[-6:-3]
    log_tau_phi = parameters[-3:]
    parameters_matrix = parameters[:-6].reshape(cfg.data_systems.n_systems,-1)

    sigma_obs = cfg.data_systems.obser_noise
    
    # hyperprior mean
    mu01 = jnp.log(cfg.hyperprior.mean.hyperprior_k_prior)
    mu02 = cfg.hyperprior.mean.hyperprior_z0_prior
    mu03 = cfg.hyperprior.mean.hyperprior_zdot_prior
    hyperprior_mean = jnp.array([mu01,mu02,mu03])
    
    hyper_mean_tau_theta = cfg.hyperprior.mean.hyperprior_mean_tau_theta
    hyper_mean_tau_z = cfg.hyperprior.mean.hyperprior_mean_tau_z
    
    # hyperprior variance
    hyper_variance_mean = jnp.log(jnp.array([cfg.hyperprior.variance.hyperprior_variance_mu_theta,
                                             cfg.hyperprior.variance.hyperprior_variance_mu_z0,
                                             cfg.hyperprior.variance.hyperprior_variance_mu_zdot]))
    hyper_variance_tau = jnp.array([cfg.hyperprior.variance.hyperprior_variance_tau_theta,
                                    cfg.hyperprior.variance.hyperprior_variance_tau_z0,
                                    cfg.hyperprior.variance.hyperprior_variance_tau_zdot])
    
    # log p(phi)
    log_phi_prior_theta = stats.norm.logpdf(mu_phi[:1],hyperprior_mean[:1],jnp.sqrt(hyper_mean_tau_theta))
    log_phi_prior_z = stats.norm.logpdf(mu_phi[-2:],hyperprior_mean[-2:],jnp.sqrt(hyper_mean_tau_z))
    log_tau_prior_theta = stats.norm.logpdf(log_tau_phi[:1],hyper_variance_mean[:1],jnp.sqrt(hyper_variance_tau[:1]))
    log_tau_prior_z = stats.norm.logpdf(log_tau_phi[-2:],hyper_variance_mean[-2:],jnp.sqrt(hyper_variance_tau[-2:]))
    
    # terms for multiplication
    log_theta_posterior = vmap_batched_log_losterior_per_system(
                                parameters_matrix,y_obser,log_tau_phi,sigma_obs,mu_phi,params_alpha, alpha_apply_fn)
    
    return log_phi_prior_theta.sum() + log_phi_prior_z.sum() + log_tau_prior_theta.sum() + log_tau_prior_z.sum() + log_theta_posterior


def log_proposal(theta_star,theta, C, R, e, params_alpha, alpha_apply_fn,y_obser):
    
    grad = jax.grad(log_posterior)(theta,params_alpha, alpha_apply_fn,y_obser)
    diff = theta_star - theta - 0.5 * e ** 2 * C @ grad
    return -0.5 * (diff.T / e**2 @ jax.scipy.linalg.cho_solve((R.T,False),diff))
    
    
def single_MALA(parameters,e,rng_key,C,R,params_alpha,alpha_apply_fn,y_obser):
        
    key, uniform_key = jax.random.split(rng_key, 2)
        
    # generate random term
    W = normal_like_tree(key,parameters)
    
    # compute gradient of log posterior
    grad = jax.grad(log_posterior)(parameters,params_alpha,alpha_apply_fn,y_obser)
        
    # new_parameters = parameters + 0.5 * e ** 2 * C @ grad + 0.5 * e ** 2 * (parameters.shape[0] + 1) / args.num_chains * (parameters - parameters.mean()) + e * R @ W
    # parameters = new_parameters
    new_parameters = parameters + 0.5 * e ** 2 * C @ grad + e * R @ W
    
    # metropolis step
    log_accept_prob = log_posterior(new_parameters,params_alpha,alpha_apply_fn,y_obser) - log_posterior(parameters,params_alpha,alpha_apply_fn,y_obser) + \
                    log_proposal(parameters,new_parameters,C,R,e,params_alpha, alpha_apply_fn,y_obser) - log_proposal(new_parameters,parameters,C,R,e,params_alpha, alpha_apply_fn,y_obser)
    
    accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
    accept = (jax.random.uniform(uniform_key) < accept_prob)
    parameters = ifelse(accept, new_parameters, parameters)
    
    # adapt step size
    e = e * jnp.sqrt(1 + cfg.langevin_sampler.step_size_lr * (accept_prob - cfg.langevin_sampler.accept_ratio))
    
    # e = jnp.minimum(e * 1.0002, 0.1)
    
    return parameters, e


def ensemble_MALA(batched_parameters,batched_e,rng_key,C,R,params_alpha,alpha_apply_fn,y_obser):
        
    keys = jax.random.split(rng_key, cfg.langevin_sampler.n_chains)
    # batch_size = int(args.num_chains // args.batch_num)
    # keys = jax.random.split(rng_key, batch_size)
    
    return vmap(single_MALA,in_axes=(0,0,0,None,None,None,None,None))(batched_parameters,batched_e,keys,C,R,params_alpha,alpha_apply_fn,y_obser)



def inference_loop(rng_key,batched_parameters,batched_e,params_alpha, alpha_apply_fn,y_obser,step,C,R):
    
    for _ in range(cfg.langevin_sampler.n_samples):
        
        # for i in range(args.batch_num):
        #     key, rng_key = jax.random.split(rng_key,2)

        #     batch_size = int(args.num_chains // args.batch_num)
        #     batched_parameters_batch, batched_e_batch = step(batched_parameters[i*batch_size:(i+1)*batch_size,:],batched_e[i*batch_size:(i+1)*batch_size,:],key,C,R,mlp_params)
        #     batched_parameters = batched_parameters.at[i*batch_size:(i+1)*batch_size,:].set(batched_parameters_batch)
        #     batched_e = batched_e.at[i*batch_size:(i+1)*batch_size,:].set(batched_e_batch)
        
        key, rng_key = jax.random.split(rng_key,2)
        batched_parameters, batched_e = step(batched_parameters,batched_e,key,C,R,params_alpha,alpha_apply_fn,y_obser)
        
        C = jnp.cov(batched_parameters.T)
        R = jnp.linalg.cholesky(C + 0.000001 * jnp.eye(3 * (cfg.data_systems.n_systems+2)))
        # R = jnp.linalg.cholesky(C)
    
    return batched_parameters, batched_e, C, R


def vmap_batch_mc_expectation(batched_parameters,params_alpha,alpha_apply_fn,y_obser):
    
    return vmap(log_posterior,in_axes=(0,None,None,None))(batched_parameters,params_alpha, alpha_apply_fn,y_obser).mean()


def loss_function(parameters,params_alpha, alpha_apply_fn,y_obser):
    
    return - vmap_batch_mc_expectation(parameters,params_alpha, alpha_apply_fn,y_obser)


def l2_error(params_alpha, alpha_apply_fn):
    
    x = jnp.linspace(-6,6,100).reshape(-1,1)
    
    y_true = cfg.true_parameters.true_a * x + cfg.true_parameters.true_b * x ** 3
    y_pred = alpha_apply_fn(params_alpha,x)
    
    return jnp.linalg.norm(y_pred - y_true, ord=2)


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
    parameter_matrix = generate_parameter_set(cfg,key)
    save_path = os.path.join(checkpoint_dir, "parameters.npy")
    jnp.save(save_path,parameter_matrix)
        
    key, rng_key = jax.random.split(rng_key,2)
    # simulations, velocity_path = vmap_batched_simulator(parameter_matrix)
    y_obser = obtain_observations(parameter_matrix, key)
    # print("     Maximum velocity: ", velocity_path.max())
    # print("     Minimum velocity: ", velocity_path.min())
    print("Done.")
    
    print("Initialising batched parameters and langevin chain...")
    key, rng_key = jax.random.split(rng_key,2)
    batched_parameters = vmap_single_chain_initialisation(
                                key, cfg).reshape(cfg.langevin_sampler.n_chains,-1)
    C = jnp.cov(batched_parameters.T)
    # R = jnp.linalg.cholesky(C + 0.000001 * jnp.eye(3 * (cfg.data_systems.n_systems+2)))
    R = jnp.linalg.cholesky(C)
    batched_e = jnp.tile(jnp.array([cfg.langevin_sampler.step_size]),(cfg.langevin_sampler.n_chains,1))
    whole_chain = jnp.empty((3, 0, batched_parameters.shape[-1]))
    print("Done.")
    
    print("Initialising alpha model...")
    alpha_key, rng_key = random.split(rng_key, 2)

    alpha_model = MLP(cfg)
    params_alpha = alpha_model.init(alpha_key, jnp.ones((1, 1))) 
    lr = optax.exponential_decay(
        cfg.models.mlp_alpha.learning_rate,
        cfg.models.mlp_alpha.n_decay_steps,
        cfg.models.mlp_alpha.decay_rate,
        staircase=True
    )
    # alpha_optimiser = optax.adamw(learning_rate=lr, weight_decay=cfg.models.mlp_alpha.weight_decay)
    alpha_optimiser = optax.adam(lr)
    opt_state_alpha = alpha_optimiser.init(params_alpha)
    alpha_apply_fn = alpha_model.apply
    step = jax.jit(ensemble_MALA,static_argnames=['alpha_apply_fn'])
    print("Done.")
    
    print("Starting inference...")
    best_l2_error = float('inf')
    log_likelihood_loss = []
    alpha_loss = []
    
    for epoch in range(10000):
        key, rng_key = jax.random.split(rng_key,2)
    
        start_time = time.time()
    
        batched_parameters, batched_e, C, R = inference_loop(key,batched_parameters,batched_e,params_alpha,alpha_apply_fn,y_obser,step,C,R)
        whole_chain = jnp.concatenate([whole_chain, batched_parameters[0:3, None, :]], axis=1)
    
        loss, grad = jax.value_and_grad(loss_function,argnums=(1))(batched_parameters,params_alpha, alpha_apply_fn,y_obser)
        updates, opt_state_alpha = alpha_optimiser.update(grad, opt_state_alpha, params_alpha)
        params_alpha = optax.apply_updates(params_alpha, updates)
    
        end_time = time.time()
        
        print(f"Epoch {epoch} completed in {end_time - start_time:.3f} seconds.")
        error = l2_error(params_alpha, alpha_apply_fn)
        
        alpha_loss.append(error)
        log_likelihood_loss.append(loss)
        
        log_data = {
            "Loss/Log Likelihood Loss": loss,
            "Loss/Alpha Function Loss": error,
            "Hyperprior/mu_k": jnp.exp(batched_parameters[0][-6]),
            "Hyperprior/mu_z0": batched_parameters[0][-5],
            "Hyperprior/mu_zdot": batched_parameters[0][-4],
            "Hyperprior/tau_k": jnp.exp(batched_parameters[0][-3]),
            "Hyperprior/tau_z0": jnp.exp(batched_parameters[0][-2]),
            "Hyperprior/tau_zdot": jnp.exp(batched_parameters[0][-1]),
        }
        
        if epoch % 100 == 0 and epoch > 0:
            fig_force, ax_force = plt.subplots(figsize=(8, 6))
            x_plot = jnp.linspace(-6, 6, 100).reshape(-1, 1)
            y_true_plot = cfg.true_parameters.true_a * x_plot + cfg.true_parameters.true_b * x_plot ** 3
            y_pred_plot = alpha_model.apply(params_alpha, x_plot)
            ax_force.plot(x_plot, y_true_plot, label='true')
            ax_force.plot(x_plot, y_pred_plot, label='predict', linestyle='--')
            ax_force.legend()
            ax_force.set_title(f'Alpha Function Fit @ Epoch {epoch}')
            ax_force.grid(True)
            log_data["Plots/Alpha Function Fit"] = wandb.Image(fig_force)
            plt.close(fig_force)
            
        wandb.log(log_data, step=epoch)
        
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
            print(f"Epoch {epoch}: New best model saved with L2 Error: {best_l2_error:.6f}")

        if epoch % 200 == 0 and epoch > 0:
            current_params = {'alpha': params_alpha}
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=current_params,
                step=1, 
                prefix='current_model_',
                overwrite=True 
            )
            print(f"Epoch {epoch}: New current model saved with L2 Error: {error:.6f}")
            
            
        if epoch % 500 == 0 and epoch > 0:
            save_path_chain = os.path.join(checkpoint_dir, "whole_chain.npy")
            save_path_upper = os.path.join(checkpoint_dir, "log_likelihood_loss.npy")
            save_path_alpha = os.path.join(checkpoint_dir, "alpha_loss.npy")
            np.save(save_path_chain, np.array(whole_chain))
            np.save(save_path_upper, np.array(log_likelihood_loss))
            np.save(save_path_alpha, np.array(alpha_loss))
            print(f"Saved whole_chain and loss snapshot at epoch {epoch}")
            
        if epoch == 9999:
            save_path_batched_parameters = os.path.join(checkpoint_dir, "batched_parameters.npy")
            save_path_batched_e = os.path.join(checkpoint_dir, "batched_e.npy")
            save_path_C = os.path.join(checkpoint_dir, "C.npy")
            save_path_R = os.path.join(checkpoint_dir, "R.npy")
            np.save(save_path_batched_parameters, batched_parameters)
            np.save(save_path_batched_e, batched_e)
            np.save(save_path_C, C)
            np.save(save_path_R, R)

        print("Epoch: ", epoch, " Log likelihood loss: ", loss, " L2 Error: ", error)
        mean_of_chain = batched_parameters.mean(axis=0)
        print("      k = ", jnp.exp(mean_of_chain)[-6], 
              "initial position and velocity = ", mean_of_chain[-5:-3])
        print("      variance = ", jnp.exp(mean_of_chain)[-3:])

        
if __name__ == "__main__":
    
    cfg = load_config(path='config_solver.yml')
    main(cfg)

        
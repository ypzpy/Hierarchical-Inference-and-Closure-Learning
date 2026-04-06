import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from functools import partial
import matplotlib.pyplot as plt
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
from constant_PINNs_nonhierarchy import *
from mlp import *
from langevin_PINNs_nonhierarchy import *
from losses_PINNs_nonhierarchy import *
from utils_nonhierarchy import *
from jax import checkpoint as remat


def bilevel_train_step(params_alpha, params_beta, opt_state_alpha, opt_state_beta, 
                        theta_batch, rng_key, beta_optimiser, alpha_optimiser, 
                        alpha_apply_fn, beta_apply_fn, y_obser):

    def inner_loop_fn(p_alpha, p_beta, opt_state_beta, rng_key):
        loss_and_grad_beta = jax.value_and_grad(pinn_physics_loss, argnums=0)

        rng_key, subkey = jax.random.split(rng_key)
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=theta_batch.shape[0]-1)
        sampled_theta = theta_batch[idx, :-6].reshape(-1, 3)

        def loop_body(carry, i):
            p_beta, opt_state_beta = carry
            inner_loss_val, grads_beta = loss_and_grad_beta(
                p_beta, p_alpha, sampled_theta, alpha_apply_fn, beta_apply_fn
            )
            update_beta, opt_state_beta = beta_optimiser.update(grads_beta, opt_state_beta, p_beta)
            p_beta = optax.apply_updates(p_beta, update_beta)
            return (p_beta, opt_state_beta), inner_loss_val

        (p_beta, opt_state_beta), inner_losses = lax.scan(
            loop_body, (p_beta, opt_state_beta), jnp.arange(cfg.models.pinn_beta.fno_steps)
        )
        return p_beta, opt_state_beta, inner_losses[-1]

    def get_alpha_loss_for_grad(p_alpha, p_beta_initial, opt_state_beta_initial, full_theta_batch, rng_key_for_inner_loop):
        
        inner_loop_fn_checkpointed = remat(inner_loop_fn)
        p_beta_final, opt_state_beta, inner_loss_val = inner_loop_fn_checkpointed(p_alpha, p_beta_initial, opt_state_beta_initial, rng_key_for_inner_loop)
        
        theta_micro_batches = full_theta_batch.reshape((cfg.langevin_sampler.batch_num, batch_size, full_theta_batch.shape[1]))

        def outer_scan_body(total_loss_so_far, micro_batch):
            micro_loss = alpha_loss_function(micro_batch, y_obser, p_beta_final, beta_apply_fn)
            new_total_loss = total_loss_so_far + micro_loss
            return new_total_loss, micro_loss

        outer_scan_body_checkpointed = remat(outer_scan_body)

        total_loss, losses_per_batch = lax.scan(
            outer_scan_body_checkpointed, 
            0.0, # initial value for total_loss_so_far
            theta_micro_batches
        )
        
        return total_loss / cfg.langevin_sampler.batch_num, losses_per_batch, p_beta_final, opt_state_beta, inner_loss_val

    outer_loss_grad_fn = jax.grad(lambda *args: get_alpha_loss_for_grad(*args)[0], argnums=0)
    
    outer_loss_val, _, params_beta, opt_state_beta, inner_loss_val = get_alpha_loss_for_grad(
        params_alpha, params_beta, opt_state_beta, theta_batch, rng_key
    )

    grads_alpha = outer_loss_grad_fn(
        params_alpha, params_beta, opt_state_beta, theta_batch, rng_key
    )
    
    updates_alpha, opt_state_alpha = alpha_optimiser.update(grads_alpha, opt_state_alpha, params_alpha)
    params_alpha = optax.apply_updates(params_alpha, updates_alpha)
    
    # params_beta, opt_state_beta, inner_loss_val = inner_loop_fn(params_alpha, params_beta, opt_state_beta, rng_key)
    
    return params_alpha, params_beta, opt_state_alpha, opt_state_beta, jnp.mean(outer_loss_val), inner_loss_val


def l2_error(params_alpha,alpha_apply_fn):
    
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

    step = jax.jit(ensemble_MALA,static_argnames=['beta_apply_fn'])
    print("Done.")

    print("Initialising alpha and beta models...")
    PINN_key, alpha_key, rng_key = random.split(rng_key, 3)

    PINN_model = PINN(cfg)
    params_beta = PINN_model.init_params(PINN_key)
    lr_beta = optax.exponential_decay(
        cfg.models.pinn_beta.learning_rate,
        cfg.models.pinn_beta.n_decay_steps,
        cfg.models.pinn_beta.decay_rate,
        staircase=True
    )
    beta_optimiser = optax.adam(lr_beta)
    opt_state_beta = beta_optimiser.init(params_beta)
    beta_apply_fn = PINN_model.forward

    alpha_model = MLP(cfg)
    params_alpha = alpha_model.init(alpha_key, jnp.ones((1, 1))) 
    lr_alpha = optax.exponential_decay(
        cfg.models.mlp_alpha.learning_rate,
        cfg.models.mlp_alpha.n_decay_steps,
        cfg.models.mlp_alpha.decay_rate,
        staircase=True
    )
    # alpha_optimiser = optax.adamw(learning_rate=lr, weight_decay=cfg.models.mlp_alpha.weight_decay)
    alpha_optimiser = optax.adam(lr_alpha)
    opt_state_alpha = alpha_optimiser.init(params_alpha)
    alpha_apply_fn = alpha_model.apply
    print("Done.")

    # Bilevel training loop
    print("Starting bilevel training...")
    best_l2_error = float('inf')
    outer_losses = []
    inner_losses = []
    alpha_losses = []
    
    jitted_bilevel_step = jax.jit(bilevel_train_step, static_argnames=["beta_optimiser",'alpha_optimiser',"alpha_apply_fn","beta_apply_fn"])
    
    for epoch in range(10000):
        start_time = time.time()
        key, key1, rng_key = random.split(rng_key,3)
        
        batched_parameters, batched_e, C, R = inference_loop(
                key,batched_parameters,batched_e,params_beta,step,C,R,y_obser,beta_apply_fn)
        whole_chain = jnp.concatenate([whole_chain, batched_parameters[0:3, None, :]], axis=1)
    
        params_alpha, params_beta, opt_state_alpha, opt_state_beta, outer_loss_val, inner_loss_val = jitted_bilevel_step(
                            params_alpha, params_beta, opt_state_alpha, opt_state_beta, 
                            batched_parameters, key1, beta_optimiser, alpha_optimiser, alpha_apply_fn, 
                            beta_apply_fn, y_obser
                        )
        end_time = time.time()
        print(f"Epoch {epoch} completed in {end_time - start_time:.5f} seconds.")
        error = l2_error(params_alpha, alpha_apply_fn)
        outer_losses.append(outer_loss_val)
        alpha_losses.append(error)
        inner_losses.append(inner_loss_val)
        
        # Log losses and parameters into wandb
        log_data = {
            "Loss/Log Likelihood Loss": outer_loss_val,
            "Loss/Alpha Function Loss": error,
            "Loss/Physics Residual": inner_loss_val,
            "Hyperprior/k_0": jnp.exp(batched_parameters[0][0]),
            "Hyperprior/z0_0": batched_parameters[0][1],
            "Hyperprior/zdot_0": batched_parameters[0][2],
            "Hyperprior/k_1": jnp.exp(batched_parameters[0][3]),
            "Hyperprior/z0_1": batched_parameters[0][4],
            "Hyperprior/zdot_1": batched_parameters[0][5],
        }
        
        # Log alpha function
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
            
            def predict_trajectory(time_array, p_vec):
                return jax.vmap(lambda _t: beta_apply_fn(params_beta, _t, p_vec))(time_array)

            predict_batch = jax.vmap(predict_trajectory, in_axes=(None, 0))

            out = predict_batch(t, parameter_matrix)

            out_pinn = out[..., None] 

            simulations, velocity_path = vmap_batched_simulator(parameter_matrix)

            # Plots for Systems 0 and 11
            fig_traj_0, axs_traj_0 = plt.subplots(1, 2, figsize=(12, 5))
            axs_traj_0[0].plot(simulations[0][:], label='True')
            axs_traj_0[0].plot(out_pinn[0, :, 0], label='Predicted')
            axs_traj_0[0].set_title("Position (System 0)")
            axs_traj_0[0].legend()
            axs_traj_0[1].plot(simulations[11][:], label='True')
            axs_traj_0[1].plot(out_pinn[11, :, 0], label='Predicted')
            axs_traj_0[1].set_title('Velocity (System 0)')
            axs_traj_0[1].legend()
            fig_traj_0.tight_layout()
            log_data["Plots/Trajectory Comparison (System 0)"] = wandb.Image(fig_traj_0)
            plt.close(fig_traj_0)
            
        wandb.log(log_data, step=epoch)
        
        # save the best model based on L2 error
        if error < best_l2_error:
            best_l2_error = error
            wandb.summary['best_l2_error'] = best_l2_error
            
            best_params = {'alpha': params_alpha, 'beta': params_beta}
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=best_params,
                step=1, 
                prefix='best_model_',
                overwrite=True 
            )
            print(f"Epoch {epoch}: New best model saved with L2 Error: {best_l2_error:.6f}")
            
        if epoch % 200 == 0 and epoch > 0:
            current_params = {'alpha': params_alpha, 'beta': params_beta}
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
            save_path_outer = os.path.join(checkpoint_dir, "outer_loss.npy")
            save_path_inner = os.path.join(checkpoint_dir, "inner_loss.npy")
            save_path_alpha = os.path.join(checkpoint_dir, "alpha_loss.npy")
            np.save(save_path_chain, np.array(whole_chain))
            np.save(save_path_outer, np.array(outer_losses))
            np.save(save_path_inner, np.array(inner_losses))
            np.save(save_path_alpha, np.array(alpha_losses))
            print(f"Saved whole_chain and loss snapshot at epoch {epoch}")
         

        print("Epoch: ", epoch, " Log likelihood loss: ", outer_loss_val, " L2 Error: ", error)
        mean_of_chain = batched_parameters.mean(axis=0)
        print("      k = ", jnp.exp(mean_of_chain)[0], 
              "initial position and velocity = ", mean_of_chain[1:3])


if __name__ == "__main__":
    
    cfg = load_config(path='config_PINNs_nonhierarchy.yml')
    main(cfg)






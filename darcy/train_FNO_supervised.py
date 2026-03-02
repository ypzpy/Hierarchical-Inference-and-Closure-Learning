import numpy as np
import os
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, value_and_grad, lax
from jax import checkpoint as remat
from jaxopt import FixedPointIteration
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
from fno import *
from utils import *
from data_generation import *
from constant_FNO_supervised import *
from mlp import *
from langevin_FNO import *
from losses_FNO import *


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
        
        # res = jnp.zeros((nx,ny))
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
        "damping": 0.8,
    }
    # --- solve PDE via fixed-point iteration ---
    u_sol, _ = solver_train.run(u_init, parameters, mlp_params, static_params)
    
    return u_sol


def vmap_poisson_train(parameter_matrix, u_inits, params_alpha, alpha_apply_fn):
    
    return vmap(G_poisson_train, in_axes=(0, 0, None, None))(parameter_matrix, u_inits, params_alpha, alpha_apply_fn)


def supervised_loss(params_beta, params_alpha, theta_batch, alpha_apply_fn, beta_apply_fn):
    # --- Forward simulations using the FNO model ---
    inputs = jnp.tile(theta_batch[:, None, None, :], (1, nx-2, ny-2, 1))
    inputs = jnp.concatenate([inputs, grid_tiled[:10,1:-1,1:-1,:]], axis=-1)
    batched_zs = jnp.mean(inputs[:, :, :, 0:cfg.data_systems.no_parameters],axis=(1,2))
    out = beta_apply_fn(params_beta, inputs).squeeze(-1)
    
    pred = jnp.zeros((theta_batch.shape[0], nx, ny))
    pred = pred.at[:,1:-1,1:-1].set(out)

    GTs = vmap_poisson_train(batched_zs, jnp.zeros((batched_zs.shape[0], nx, ny)), params_alpha, alpha_apply_fn)
    supervised_loss = jnp.linalg.norm(pred - GTs) ** 2 / GTs.size

    return supervised_loss / theta_batch.shape[0]
    # --- Forward simulations using the FNO model ---
    # inputs = jnp.tile(theta_batch[:, None, None, :], (1, nx, ny, 1))
    # inputs = jnp.concatenate([inputs, grid_tiled], axis=-1)
    # batched_zs = jnp.mean(inputs[:, :, :, 0:cfg.data_systems.no_parameters],axis=(1,2))
    # out = beta_apply_fn(params_beta, inputs).squeeze(-1)

    # GTs = vmap_poisson_train(batched_zs, jnp.zeros((batched_zs.shape[0],nx,ny)), params_alpha, alpha_apply_fn)
    # supervised_loss = jnp.linalg.norm(out - GTs) ** 2 / GTs.size

    # return supervised_loss / theta_batch.shape[0]


def bilevel_train_step(params_alpha, params_beta, opt_state_alpha, opt_state_beta, 
                        theta_batch, rng_key, beta_update_fn, alpha_optimiser, 
                        alpha_apply_fn, beta_apply_fn, y_obser, H_mats):

    def inner_loop_fn(p_alpha, p_beta, opt_state_beta, rng_key):
        loss_and_grad_beta = jax.value_and_grad(supervised_loss, argnums=0)

        rng_key, subkey = jax.random.split(rng_key)
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=theta_batch.shape[0]-1)
        sampled_theta = theta_batch[idx, :-2*cfg.data_systems.no_parameters].reshape(-1, cfg.data_systems.no_parameters)

        def loop_body(carry, i):
            p_beta, opt_state_beta, rng_key = carry
            rng_key, subkey = jax.random.split(rng_key)
            idx = jax.random.randint(subkey, shape=(10,), minval=0, maxval=theta_batch.shape[0]-1)
            inner_loss_val, grads_beta = loss_and_grad_beta(
                p_beta, p_alpha, sampled_theta[idx], alpha_apply_fn, beta_apply_fn
            )
            p_beta, opt_state_beta = beta_update_fn(grads_beta, p_beta, opt_state_beta)
            return (p_beta, opt_state_beta, rng_key), inner_loss_val
        
        rng_key, subkey = jax.random.split(rng_key)
        (p_beta, opt_state_beta, subkey), inner_losses = lax.scan(
            loop_body, (p_beta, opt_state_beta, subkey), jnp.arange(cfg.models.fno_beta.fno_steps)
        )
        return p_beta, opt_state_beta, inner_losses[-1]

    def get_alpha_loss_for_grad(p_alpha, p_beta_initial, opt_state_beta_initial, full_theta_batch, rng_key_for_inner_loop):
        
        inner_loop_fn_checkpointed = remat(inner_loop_fn)
        p_beta_final, _, _ = inner_loop_fn_checkpointed(p_alpha, p_beta_initial, opt_state_beta_initial, rng_key_for_inner_loop)
        
        theta_micro_batches = full_theta_batch.reshape((cfg.langevin_sampler.batch_num, batch_size, full_theta_batch.shape[1]))

        def outer_scan_body(total_loss_so_far, micro_batch):
            micro_loss = alpha_loss_function(micro_batch, H_mats, y_obser, p_beta_final, beta_apply_fn)
            new_total_loss = total_loss_so_far + micro_loss
            return new_total_loss, micro_loss

        outer_scan_body_checkpointed = remat(outer_scan_body)

        total_loss, losses_per_batch = lax.scan(
            outer_scan_body_checkpointed, 
            0.0, # initial value for total_loss_so_far
            theta_micro_batches
        )
        
        return total_loss / cfg.langevin_sampler.batch_num, losses_per_batch

    outer_loss_grad_fn = jax.grad(lambda *args: get_alpha_loss_for_grad(*args)[0], argnums=0)
    
    outer_loss_val, _ = get_alpha_loss_for_grad(
        params_alpha, params_beta, opt_state_beta, theta_batch, rng_key
    )

    grads_alpha = outer_loss_grad_fn(
        params_alpha, params_beta, opt_state_beta, theta_batch, rng_key
    )
    
    updates_alpha, opt_state_alpha = alpha_optimiser.update(grads_alpha, opt_state_alpha, params_alpha)
    params_alpha = optax.apply_updates(params_alpha, updates_alpha)
    
    params_beta, opt_state_beta, inner_loss_val = inner_loop_fn(params_alpha, params_beta, opt_state_beta, rng_key)
    
    return params_alpha, params_beta, opt_state_alpha, opt_state_beta, jnp.mean(outer_loss_val), inner_loss_val


def l2_error(params_alpha,alpha_apply_fn,max_value):
    
    x = jnp.linspace(0.0,max_value,40).reshape(-1,1)
    
    # y_true = x ** 2
    y_true = nonlinear_function(x)
    
    y_pred = alpha_apply_fn({'params': params_alpha},x)

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
    parameter_matrix = generate_parameter_set(cfg,key)
    systems = vmap_batched_poisson_jaxopt(parameter_matrix)
    save_path = os.path.join(checkpoint_dir, "parameters.npy")
    jnp.save(save_path,parameter_matrix)

    inputs = jnp.tile(parameter_matrix[:, None, None, :], (1, nx-2, ny-2, 1))
    inputs = jnp.concatenate([inputs, grid_tiled[:,1:-1,1:-1,:]], axis=-1)
    # inputs = jnp.tile(parameter_matrix[:, None, None, :], (1, nx, ny, 1))
    # inputs = jnp.concatenate([inputs, grid_tiled], axis=-1)

    key, rng_key = jax.random.split(rng_key,2)
    H_mats, idxs, y_obser = obtain_observations(parameter_matrix, key)
    print("     Maximum u value: ", systems.max())
    print("Done.")
    
    key, rng_key = jax.random.split(rng_key,2)
    key, rng_key = jax.random.split(rng_key,2)

    print("Initialising batched parameters and langevin chain...")
    key, rng_key = jax.random.split(rng_key,2)
    batched_parameters = vmap_single_chain_initialisation(key).reshape(cfg.langevin_sampler.n_chains,-1)
    C = jnp.cov(batched_parameters.T)
    # R = jnp.linalg.cholesky(C + 0.000001 * jnp.eye(3 * (cfg.data_systems.n_systems+2)))
    R = jnp.linalg.cholesky(C)
    batched_e = jnp.tile(jnp.array([cfg.langevin_sampler.step_size]),(cfg.langevin_sampler.n_chains,1))
    whole_chain = jnp.empty((3, 0, batched_parameters.shape[-1]))

    step = jax.jit(ensemble_MALA,static_argnames=['beta_apply_fn'])
    print("Done.")

    print("Initialising alpha and beta models...")
    fno_key, alpha_key, rng_key = random.split(rng_key, 3)

    fno_model = FNO(cfg, FNO_utils2D)
    params_beta, opt_state_beta = fno_model.init_model(fno_key, inputs[0])
    beta_update_fn = partial(fno_model.update)
    beta_apply_fn = partial(fno_model.vmap_z_call)

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
    
    print("Initialising solver...")
    pi = jnp.pi
    phi_1 = jnp.sin(2 * pi * X) * jnp.sin(2 * pi * Y)
    phi_2 = jnp.sin(2 * pi * X) * jnp.sin(pi * Y)
    phi_3 = jnp.sin(pi * X) * jnp.sin(2 * pi * Y)
    basis_functions = jnp.stack([phi_1, phi_2, phi_3], axis=0)

    global solver_train
    fixed_point_iteration_train = make_fixed_point_iteration_train(alpha_apply_fn, basis_functions, nx, ny, hx, hy)
    jitted_fixed_point_train = jax.jit(fixed_point_iteration_train)
    solver_train = FixedPointIteration(
        fixed_point_fun=jitted_fixed_point_train,
        maxiter=12,
        tol=1e-5,
        jit=True,
        implicit_diff=True
    )
    print("Done.")

    # Bilevel training loop
    print("Starting bilevel training...")
    best_l2_error = float('inf')
    lower_loss = []
    upper_loss = []
    alpha_loss = []
    
    jitted_bilevel_step = jax.jit(bilevel_train_step, static_argnames=["beta_update_fn",'alpha_optimiser',"alpha_apply_fn","beta_apply_fn"])
    
    for epoch in range(20000):
        start_time = time.time()
        key, key1, rng_key = random.split(rng_key,3)
        
        batched_parameters, batched_e, C, R = inference_loop(
                key,batched_parameters,batched_e,H_mats,params_beta,step,C,R,y_obser,beta_apply_fn)
        whole_chain = jnp.concatenate([whole_chain, batched_parameters[0:3, None, :]], axis=1)
    
        params_alpha, params_beta, opt_state_alpha, opt_state_beta, outer_loss_val, inner_loss_val = jitted_bilevel_step(
                            params_alpha, params_beta, opt_state_alpha, opt_state_beta, 
                            batched_parameters, key1, beta_update_fn, alpha_optimiser, alpha_apply_fn, 
                            beta_apply_fn, y_obser, H_mats
                        )
        end_time = time.time()
        print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds.")
        error = l2_error(params_alpha, alpha_apply_fn,systems.max())
        upper_loss.append(outer_loss_val)
        alpha_loss.append(error)
        lower_loss.append(inner_loss_val)
        
        # Log losses and parameters into wandb
        log_data = {
            "Loss/Log Likelihood Loss": outer_loss_val,
            "Loss/Alpha Function Loss": error,
            "Loss/Physics Residual": inner_loss_val,
            "Hyperprior/mu_z1": batched_parameters[0][-6],
            "Hyperprior/mu_z2": batched_parameters[0][-5],
             "Hyperprior/mu_z3": batched_parameters[0][-4],
            "Hyperprior/tau_z1": jnp.exp(batched_parameters[0][-3]),
            "Hyperprior/tau_z2": jnp.exp(batched_parameters[0][-2]),
             "Hyperprior/tau_z3": jnp.exp(batched_parameters[0][-1]),
        }
        
        # Log alpha function
        if epoch % 50 == 0 and epoch > 0:
            fig_force, ax_force = plt.subplots(figsize=(8, 6))
            x_plot = jnp.linspace(0.0,systems.max(),100).reshape(-1,1)
            y_true_plot = nonlinear_function(x_plot)
            y_pred_plot = alpha_model.apply({'params': params_alpha}, x_plot)
            ax_force.plot(x_plot, y_true_plot, label='true')
            ax_force.plot(x_plot, y_pred_plot, label='predict', linestyle='--')
            ax_force.legend()
            ax_force.set_title(f'Alpha Function Fit @ Epoch {epoch}')
            ax_force.grid(True)
            log_data["Plots/Alpha Function Fit"] = wandb.Image(fig_force)
            plt.close(fig_force)
            
            # plot FNO predictions
            out = beta_apply_fn(params_beta, inputs).squeeze(-1)
            
            pred = jnp.zeros((parameter_matrix.shape[0], nx, ny))
            pred = pred.at[:,1:-1,1:-1].set(out)

            # Plot for System 0
            slice = 0
            fig_slice, axs_slice = plt.subplots(1, 2, figsize=(10, 4))
            # Ground Truth
            im0 = axs_slice[0].contourf(systems[slice], levels=50, cmap="viridis")
            fig_slice.colorbar(im0, ax=axs_slice[0], orientation="vertical", fraction=0.046, pad=0.04)
            axs_slice[0].set_title(f"Ground Truth (slice {slice})")
            axs_slice[0].set_xlabel("x")
            axs_slice[0].set_ylabel("y")

            # Prediction
            im1 = axs_slice[1].contourf(pred[slice], levels=50, cmap="viridis")
            fig_slice.colorbar(im1, ax=axs_slice[1], orientation="vertical", fraction=0.046, pad=0.04)
            axs_slice[1].set_title(f"Prediction (slice {slice})")
            axs_slice[1].set_xlabel("x")
            axs_slice[1].set_ylabel("y")

            fig_slice.tight_layout()

            # Log to wandb
            log_data[f"Plots/Slice Comparison (slice {slice})"] = wandb.Image(fig_slice)

            plt.close(fig_slice)
            
            # Plot for System 15
            slice = 15
            fig_slice2, axs_slice2 = plt.subplots(1, 2, figsize=(10, 4))
            # Ground Truth
            im0 = axs_slice2[0].contourf(systems[slice], levels=50, cmap="viridis")
            fig_slice2.colorbar(im0, ax=axs_slice2[0], orientation="vertical", fraction=0.046, pad=0.04)
            axs_slice2[0].set_title(f"Ground Truth (slice {slice})")
            axs_slice2[0].set_xlabel("x")
            axs_slice2[0].set_ylabel("y")

            # Prediction
            im1 = axs_slice2[1].contourf(pred[slice], levels=50, cmap="viridis")
            fig_slice2.colorbar(im1, ax=axs_slice2[1], orientation="vertical", fraction=0.046, pad=0.04)
            axs_slice2[1].set_title(f"Prediction (slice {slice})")
            axs_slice2[1].set_xlabel("x")
            axs_slice2[1].set_ylabel("y")

            fig_slice2.tight_layout()

            # Log to wandb
            log_data[f"Plots/Slice Comparison (slice {slice})"] = wandb.Image(fig_slice2)

            plt.close(fig_slice)
            
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
            save_path_lower = os.path.join(checkpoint_dir, "lower_loss.npy")
            save_path_upper = os.path.join(checkpoint_dir, "upper_loss.npy")
            save_path_alpha = os.path.join(checkpoint_dir, "alpha_loss.npy")
            np.save(save_path_chain, np.array(whole_chain))
            np.save(save_path_lower, np.array(lower_loss))
            np.save(save_path_upper, np.array(upper_loss))
            np.save(save_path_alpha, np.array(alpha_loss))
            print(f"Saved whole_chain and loss snapshot at epoch {epoch}")

        print("Epoch: ", epoch, " Log likelihood loss: ", outer_loss_val, " L2 Error: ", error)
        mean_of_chain = batched_parameters.mean(axis=0)
        print("      z1, z2, z3 = ", mean_of_chain[-6:-3])
        print("      variance = ", jnp.exp(mean_of_chain)[-3:])


if __name__ == "__main__":
    
    cfg = load_config(path="config_FNO_supervised.yml")
    main(cfg)






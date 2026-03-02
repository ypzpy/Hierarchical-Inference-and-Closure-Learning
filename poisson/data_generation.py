import jax
import jax.numpy as jnp
from utils import *
from constant_FNO_physics import * 
from jax.scipy.sparse.linalg import cg as jax_cg
from jaxopt import FixedPointIteration

def nonlinear_function(x):
    
    # return 0.3*x**3 - 0.4 * x**2 + 0.5*jnp.sin(3*x)
    return x ** 2 / 2


def matvec_operator(v_interior, a_full, nx, ny):
    """
    Computes the matrix-vector product A @ v without forming A explicitly.
    """
    # Embed interior unknowns into full grid with zero boundary
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


def make_fixed_point_iteration_train(basis_functions, nx, ny, hx, hy):
    def fixed_point_iteration_train(u_full, parameters, static_params):
        z_coeffs = parameters

        f_full = static_params["f_full"]
        damping = static_params["damping"]
        # basis_functions provided externally; could also read from static_params

        Z_x_sum = jnp.einsum('j,jxy->xy', z_coeffs, basis_functions)
        Z_x_field = nn.softplus(Z_x_sum)

        # g_u = u_full ** 2
        g_u = nonlinear_function(u_full)
        a_full = Z_x_field * sigmoid_fn(g_u.reshape(nx, ny))

        f_weak = rhs_weakform(f_full, hx, hy)

        matvec_fun = lambda v: matvec_operator(v, a_full, nx, ny)
        u_interior, _ = jax_cg(matvec_fun, f_weak, tol=1e-6)
        u_temp = u_full.at[1:-1, 1:-1].set(u_interior.reshape(nx-2, ny-2))

        u_new = (1.0 - damping) * u_full + damping * u_temp
        return u_new

    return fixed_point_iteration_train


def G_poisson_true_jaxopt(parameters):
    """
    Solve nonlinear Poisson equation for one system.
    parameters: (3,) array = forcing term coefficients
    return: u(x,y) solution field, shape (nx, ny)
    """
    z_coeffs = parameters  # (3,)
    f_full = f_source_term(X,Y)
    u_init = jnp.zeros((nx, ny))

    # Use true nonlinear a(u) function
    static_params = {
        "f_full": f_full,
        "damping": 0.6,
    }
    pi = jnp.pi
    phi_1 = jnp.sin(2 * pi * X) * jnp.sin(2 * pi * Y)
    phi_2 = jnp.sin(2 * pi * X) * jnp.sin(pi * Y)
    phi_3 = jnp.sin(pi * X) * jnp.sin(2 * pi * Y)
    basis_functions = jnp.stack([phi_1, phi_2, phi_3], axis=0)

    fixed_point_iteration_train = make_fixed_point_iteration_train(basis_functions, nx, ny, hx, hy)

    # JIT the inner fixed-point iteration (it uses closure nx,ny,hx,hy)
    jitted_fixed_point_train = jax.jit(fixed_point_iteration_train)

    # Let jaxopt handle JIT inside; pass jit=True so run() is compiled
    solver_train = FixedPointIteration(
        fixed_point_fun=jitted_fixed_point_train,
        maxiter=25,
        tol=1e-6,
        jit=True
    )
    # --- solve PDE via fixed-point iteration ---
    u_sol, _ = solver_train.run(u_init, parameters, static_params)
    
    return u_sol


@jit
def vmap_batched_poisson_jaxopt(batched_parameters):
    """
    parameters: (batch, 3)
    returns: (batch, nx, ny) PDE solutions
    """
    return vmap(G_poisson_true_jaxopt)(batched_parameters)


def generate_observation_matrices(key, num_systems, nx, ny, obs_fraction=0.05):
    """
    Generate different observation matrices H for each system.
    
    Args:
        key: jax.random.PRNGKey
        num_systems: number of systems (e.g. 20)
        nx, ny: grid dimensions (e.g. 50, 50)
        obs_fraction: fraction of observed points (e.g. 0.05 for 5%)
    
    Returns:
        H_matrices: list of (n_obs, nx*ny) observation matrices
        obs_indices: list of index arrays of observed points
    """
    N = nx * ny
    n_obs = int(N * obs_fraction)
    
    H_matrices = []
    obs_indices = []
    
    keys = jax.random.split(key, num_systems)
    
    for k in keys:
        idx = jax.random.choice(k, N, shape=(n_obs,), replace=False)
        obs_indices.append(idx)
        
        # Build sparse selection matrix H
        H = jnp.zeros((n_obs, N))
        H = H.at[jnp.arange(n_obs), idx].set(1.0)
        
        H_matrices.append(H)
        
    H_matrices = jnp.stack(H_matrices, axis=0)      # (num_systems, n_obs, N)
    obs_indices = jnp.stack(obs_indices, axis=0) 
    
    return H_matrices, obs_indices


def obtain_observations(parameter_matrix, rng_key):
    """
    Generate noisy observations for each system.
    
    Args:
        sim_data: (num_systems, nx, ny) true field
        H_mats: list of observation matrices (n_obs, nx*ny)
        obs_indices: list of index arrays
        key: PRNG key
        noise_std: std of Gaussian noise
    
    Returns:
        obs_data: (num_systems, n_obs) noisy observations
    """
    
    sim_data = vmap_batched_poisson_jaxopt(parameter_matrix)  # (num_systems, nx, ny)
    noise_std = cfg.data_systems.obser_noise
    
    key, rng_key = jax.random.split(rng_key)
    H_mats, idxs = generate_observation_matrices(key, num_systems=cfg.data_systems.n_systems, nx=nx, ny=ny, obs_fraction=cfg.data_systems.obser_fraction)

    num_systems, _, _ = sim_data.shape
    n_obs = H_mats[0].shape[0]
    obs_data = jnp.zeros((num_systems, n_obs))

    keys = jax.random.split(rng_key, num_systems)
    
    for i in range(num_systems):
        u_flat = sim_data[i].ravel()
        true_obs = H_mats[i] @ u_flat  # (n_obs,)
        noise = noise_std * jax.random.normal(keys[i], shape=true_obs.shape)
        obs_data = obs_data.at[i].set(true_obs + noise)
    
    return H_mats, idxs, obs_data


def single_chain_initialisation(rng_key):
    
    single_chain = jnp.zeros((cfg.data_systems.n_systems+2)*cfg.data_systems.no_parameters)

    rng_key,key1,key2,key3,key4,key5,key6 = jax.random.split(rng_key,7)
        
    z1_mu = jax.random.normal(key1)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z)+cfg.hyperprior.mean.hyperprior_z1_prior
    z2_mu = jax.random.normal(key2)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z)+cfg.hyperprior.mean.hyperprior_z2_prior
    z3_mu = jax.random.normal(key3)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z)+cfg.hyperprior.mean.hyperprior_z3_prior

    z1_tau = jnp.exp(jax.random.normal(key4)*jnp.sqrt(cfg.hyperprior.variance.hyperprior_variance_tau_z1)+jnp.log(cfg.hyperprior.variance.hyperprior_variance_mu_z1))
    z2_tau = jnp.exp(jax.random.normal(key5)*jnp.sqrt(cfg.hyperprior.variance.hyperprior_variance_tau_z2)+jnp.log(cfg.hyperprior.variance.hyperprior_variance_mu_z2))
    z3_tau = jnp.exp(jax.random.normal(key6)*jnp.sqrt(cfg.hyperprior.variance.hyperprior_variance_tau_z3)+jnp.log(cfg.hyperprior.variance.hyperprior_variance_mu_z3))

    
    for i in range(cfg.data_systems.n_systems):
        
        key,key1,key2,rng_key = jax.random.split(rng_key,4)
        single_chain = single_chain.at[i*cfg.data_systems.no_parameters].set(jax.random.normal(key)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z)+cfg.hyperprior.mean.hyperprior_z1_prior)
        single_chain = single_chain.at[i*cfg.data_systems.no_parameters+1].set(jax.random.normal(key1)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z)+cfg.hyperprior.mean.hyperprior_z2_prior)
        single_chain = single_chain.at[i*cfg.data_systems.no_parameters+2].set(jax.random.normal(key2)*jnp.sqrt(cfg.hyperprior.mean.hyperprior_mean_tau_z)+cfg.hyperprior.mean.hyperprior_z3_prior)

    single_chain = single_chain.at[-2*cfg.data_systems.no_parameters:-cfg.data_systems.no_parameters].set(jnp.array([z1_mu,z2_mu,z3_mu]))
    single_chain = single_chain.at[-cfg.data_systems.no_parameters:].set(jnp.log(jnp.array([z1_tau,z2_tau,z3_tau])))

    return single_chain


def vmap_single_chain_initialisation(rng_key):
    
    batched_rng_key = jax.random.split(rng_key, cfg.langevin_sampler.n_chains)
    # init_fn = partial(single_chain_initialisation, cfg=cfg)
    
    return vmap(single_chain_initialisation)(batched_rng_key)


def generate_parameter_set(args,rng_key):
    z1_mean = cfg.true_parameters.hyperprior_z1
    z2_mean = cfg.true_parameters.hyperprior_z2
    z3_mean = cfg.true_parameters.hyperprior_z3

    key_z1, key_z2, key_z3 = jax.random.split(rng_key, 3)

    z1_samples = jax.random.normal(key_z1, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_z1) + z1_mean
    z2_samples = jax.random.normal(key_z2, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_z2) + z2_mean
    z3_samples = jax.random.normal(key_z3, (cfg.data_systems.n_systems,)) * jnp.sqrt(cfg.true_parameters.tau_phi_z3) + z3_mean

    parameter_matrix = jnp.stack([z1_samples, z2_samples, z3_samples], axis=-1)
    # parameter_matrix = jnp.stack([z1_samples, z2_samples], axis=-1)

    return parameter_matrix
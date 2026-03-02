import jax.numpy as jnp
from ml_collections import config_dict
from utils import *

cfg = load_config("config_FNO_supervised.yml")
nx = cfg.data_systems.nx
nt = cfg.data_systems.nt
hx = 2 * cfg.data_systems.L_x / (cfg.data_systems.nx - 1)
ht = cfg.data_systems.L_t / (cfg.data_systems.nt - 1)

x = jnp.linspace(-cfg.data_systems.L_x, cfg.data_systems.L_x, cfg.data_systems.nx)
t = jnp.linspace(0, cfg.data_systems.L_t, cfg.data_systems.nt)

T_grid, X_grid = jnp.meshgrid(t, x, indexing='ij')

# Stack to form (nt, nx, 2) -> Channels: (x, t)
grid = jnp.stack([X_grid, T_grid], axis=-1) 

grid_full = grid # Shape: (nt, nx, 2)

# Tile for batch dimension 
grid_tiled = jnp.tile(grid_full[None, ...], (cfg.data_systems.n_systems, 1, 1, 1))
batch_size = cfg.langevin_sampler.n_chains // cfg.langevin_sampler.batch_num
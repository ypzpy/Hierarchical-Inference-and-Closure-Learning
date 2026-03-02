import jax.numpy as jnp
from ml_collections import config_dict
from utils import *

cfg = load_config("config_FNO_supervised.yml")
nx = cfg.data_systems.nx
ny = cfg.data_systems.ny
hx = cfg.data_systems.L_x / (cfg.data_systems.nx - 1)
hy = cfg.data_systems.L_y / (cfg.data_systems.ny - 1)

x = jnp.linspace(0, cfg.data_systems.L_x, cfg.data_systems.nx)
y = jnp.linspace(0, cfg.data_systems.L_y, cfg.data_systems.ny)
X, Y = jnp.meshgrid(x, y)
grid = jnp.stack(jnp.meshgrid(x,y, indexing='ij'))
grid_ch_last = jnp.moveaxis(grid, 0, -1)               # (nx, ny, 2)
grid_tiled = jnp.tile(grid_ch_last[None, ...], (cfg.data_systems.n_systems, 1, 1, 1))

f_full = f_source_term(X,Y)

batch_size = cfg.langevin_sampler.n_chains // cfg.langevin_sampler.batch_num
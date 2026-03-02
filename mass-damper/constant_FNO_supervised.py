import jax.numpy as jnp
from ml_collections import config_dict
from utils import *

cfg = load_config(path='config_FNO_supervised.yml')
dt = cfg.data_systems.dt
T = int(cfg.data_systems.stop_time / cfg.data_systems.dt) + 1
t = jnp.linspace(0., cfg.data_systems.stop_time, T)
forcing = 10 * jnp.sin(t)
forcing_gradient = 10 * jnp.cos(t)
batch_size = cfg.langevin_sampler.n_chains // cfg.langevin_sampler.batch_num

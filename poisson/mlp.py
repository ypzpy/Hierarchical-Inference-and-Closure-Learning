from flax import linen as nn
import jax.numpy as jnp
from ml_collections import config_dict
from jax import random, jit
import functools
from flax.core.frozen_dict import FrozenDict


class MLP(nn.Module):
    """A simple MLP for the discrepancy term h_alpha."""
    cfg: config_dict.ConfigDict

    @nn.compact
    def __call__(self, x):
        for feat in self.cfg.models.mlp_alpha.hidden_features:
            x = nn.Dense(features=feat)(x)
            x = nn.silu(x)
        x = nn.Dense(features=self.cfg.models.mlp_alpha.output_features)(x)
        return x
    
    
class PINN:
    def __init__(self, cfg):
        self.layers = cfg.models.pinn_beta.layers 

    def init_params(self, key):
        params = []
        keys = random.split(key, len(self.layers) - 1)
        for i, (in_dim, out_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            w_key, b_key = random.split(keys[i])
            # Xavier Initialization
            limit = jnp.sqrt(6.0 / (in_dim + out_dim))
            W = random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit)
            b = jnp.zeros((out_dim,))
            params.append((W, b))
        return params

    @functools.partial(jit, static_argnums=(0,))
    def forward(self, params, xy, p_vec):
        """
        xy: shape (2,) -> [x, y]
        p_vec: shape (3,) -> [p1, p2, p3]
        """
        if isinstance(params, (dict, FrozenDict)):
            params_list = []
            for layer in params.values():
                W = layer["kernel"]
                b = layer["bias"]
                params_list.append((W, b))
            params = params_list

        inputs = jnp.concatenate([xy, p_vec]) 
        x = inputs

        # hidden layers
        for W, b in params[:-1]:
            x = jnp.dot(x, W) + b
            x = jnp.tanh(x)

        # final layer
        W_last, b_last = params[-1]
        return (jnp.dot(x, W_last) + b_last)[0]
   
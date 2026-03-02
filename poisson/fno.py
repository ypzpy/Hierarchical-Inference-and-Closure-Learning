import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
from utils import activation_functions
from collections.abc import Callable
import matplotlib.pyplot as plt

class FNO_utils1D():
    @staticmethod
    def RMult(R, f):
        # R   [modes, ch_in, ch_out]
        # f   [modes, ch_in]
        # out [modes, ch_out]
        f = f[:R.shape[0], ...]
        # print('R.shape, f.shape:', R.shape, f.shape)
        return jnp.einsum('xio,xi->xo', R, f)
    @staticmethod
    def fftpad(v, Rf):
        return (v * 0.).at[:Rf.shape[0], :Rf.shape[1]].set(Rf)
    @staticmethod
    def get_conv(v):
        return nn.Conv(features = v.shape[-1], kernel_size=(1,), strides=(1,), padding='SAME')
    @staticmethod
    def get_shape_R(n_modes, v):
        return (n_modes, v.shape[-1], v.shape[-1])
    @staticmethod
    def get_fft_axes():
        return (0,)

class FNO_utils2D():
    @staticmethod
    def RMult(R, f):
        # R   [modes, modes, ch_in, ch_out]
        # f   [modes, modes, ch_in]
        # out [modes, modes, ch_out]
        # f = f[ :R.shape[0], :R.shape[0], ...]
        # # print('R.shape, f.shape:', R.shape, f.shape)
        # return jnp.einsum('xyio,xyi->xyo', R, f)
        nx = min(R.shape[0], f.shape[0])
        ny = min(R.shape[1], f.shape[1])
        f = f[:nx, :ny, ...]
        R = R[:nx, :ny, ...]
        return jnp.einsum('xyio,xyi->xyo', R, f)
    @staticmethod
    def fftpad(v, Rf):
        return (v * 0.).at[:Rf.shape[0], :Rf.shape[1], :Rf.shape[2]].set(Rf)
    @staticmethod
    def get_conv(v):
        return nn.Conv(features = v.shape[-1], kernel_size=(1,1), strides=(1,1), padding='SAME')
    @staticmethod
    def get_shape_R(n_modes, v):
        return (n_modes, n_modes, v.shape[-1], v.shape[-1])
    @staticmethod
    def get_fft_axes():
        return (0,1)

class FLayer(nn.Module):
    n_modes: int
    FNO_utils: object #FNO_utils1D

    def complex_kernel_init(rng, shape):
        key1, key2 = random.split(rng)
        x = jax.lax.complex(random.uniform(key1, shape),  random.uniform(key2, shape))
        return x/(shape[-1]**2.)

    complex_kernel_init: Callable = complex_kernel_init

    @nn.compact
    def __call__(self, v):
        # print('### START FNO Layer ###')
        # print('v.shape: ', v.shape)

        W = self.FNO_utils.get_conv(v)(v)
        # print('W.shape', W.shape)

        axes = self.FNO_utils.get_fft_axes( )
        f = jnp.fft.rfftn(v, axes=axes)
        # print('f.shape:, f.type', f.shape)

        shape_R = self.FNO_utils.get_shape_R(self.n_modes, v)

        # print('shape_R:', shape_R)
        R = self.param('R', self.complex_kernel_init, shape_R)

        Rf = self.FNO_utils.RMult(R, f)
        # print('Rf:', Rf.shape)
        fp = self.FNO_utils.fftpad(f, Rf)
        # print('fp.shape:', fp.shape)
        vi = jnp.fft.irfftn(fp, s=tuple(v.shape[axis] for axis in axes), axes=axes)
        # print('vi.shape: ', vi.shape)

        v_out = W + vi

        # print('### END FNO Layer ###')
        return v_out


class FNO(nn.Module):
    cfg: dict
    FNO_utils: object

    '''
        cfg.models.fno_beta.dim_v
        cfg.models.fno_beta.n_modes
        cfg.models.fno_beta.out_dim
    '''

    @nn.compact
    def __call__(self, z):
        # print('\n### START ###')

        v = z
        # print('v.shape: ', v.shape)

        #! P layer
        v = nn.Dense(self.cfg.models.fno_beta.dim_v)(v)
        v = activation_functions[self.cfg.models.fno_beta.activation](v)
        v = nn.Dense(self.cfg.models.fno_beta.dim_v)(v)
        # print('post p layer v.shape: ', v.shape)

        for _ in range(self.cfg.models.fno_beta.n_layers):
            v =  FLayer(self.cfg.models.fno_beta.n_modes, FNO_utils=self.FNO_utils)(v)
            v = activation_functions[self.cfg.models.fno_beta.activation](v)

        #! Q layer
        v = nn.Dense(self.cfg.models.fno_beta.dim_v)(v)
        v = activation_functions[self.cfg.models.fno_beta.activation](v)
        v = nn.Dense(self.cfg.models.fno_beta.out_dim)(v)
        # print('post Q layer v.shape: ', v.shape)
        # print('### END ###\n')
        return v

    def vmap_z_call(self, params, z):
        return jax.vmap(self.apply, in_axes=(None, 0))(params, z)

    def init_model(self, key, z):

        rng, key = random.split(key) # PRNG Key
        output, params = self.init_with_output(key, z)
        # print('output.shape', output.shape)

        count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        # print('count', count)

        lr = optax.exponential_decay(
            self.cfg.models.fno_beta.learning_rate,
            self.cfg.models.fno_beta.n_decay_steps,
            self.cfg.models.fno_beta.decay_rate,
            staircase=True
        )
        
        if self.cfg.models.fno_beta.opt_type == 'adam':
            # print('using ADAM')
            base_optimizer = optax.adam(learning_rate=lr)
            
        if self.cfg.models.fno_beta.opt_type == 'amsgrad':
            # print('using AMSGRAD')
            base_optimizer = optax.amsgrad(learning_rate=lr)
            
        if self.cfg.models.fno_beta.opt_type == 'adamw':
            # print('using ADAMW')
            base_optimizer = optax.adamw(learning_rate=lr, weight_decay=self.cfg.models.fno_beta.weight_decay)

        if hasattr(self.cfg.models.fno_beta, 'gradient_clip'):
            # 3. If so, chain the clipping transformation with the base optimizer
            self.opt = optax.chain(
                optax.clip_by_global_norm(self.cfg.models.fno_beta.gradient_clip),
                base_optimizer
            )
        else:
            # 4. If not, just use the base optimizer
            self.opt = base_optimizer
            
        opt_state  = self.opt.init(params)

        # print(self.tabulate(key, z, compute_flops=True, compute_vjp_flops=True))

        return params, opt_state

    def update(self, grads, params, opt_state):
        # print('CONJUGATE UPDATE ONET') 
        grads = jax.tree_map(lambda x: x.conj(), grads)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
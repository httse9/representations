import optax
from flax import nnx
import jax.numpy as jnp


class Encoder(nnx.Module):
    
    def __init__(self, obs_dim: int, feat_dim: int, eig_dim: int, duals_initial_val: float, barrier_initial_val: float, 
                obs_type: str, rngs: nnx.Rngs): 
        
        self.obs_type = obs_type
        
        if obs_type == 'image':
            
            obs_dim = feat_dim
            
            self.eig_conv = nnx.Sequential(*[
                nnx.Conv(1, 16, kernel_size=(8, 8), strides = 4, rngs=rngs),
                nnx.relu,
                nnx.Conv(16, 16, kernel_size=(4, 4), strides = 2, rngs=rngs),
                nnx.relu,
                nnx.Conv(16, 16, kernel_size=(3, 3), strides = 2, rngs=rngs),
                nnx.relu
            ])
            
            # todo - find a neat way to do this
            self.reshaper_linear = nnx.Linear(144, obs_dim, rngs=rngs)
            
                
        self.eig_linear = nnx.Sequential(*[
            nnx.Linear(obs_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, eig_dim, rngs=rngs),
        ])
        
        self.duals = nnx.Param(jnp.tril(duals_initial_val * jnp.ones((eig_dim, eig_dim))))
        self.barrier_coefs = nnx.Param(barrier_initial_val * jnp.ones((1, 1)))

    def __call__(self, obs):
        
        if self.obs_type == 'image':
            
            obs = self.eig_conv(obs)
            obs = jnp.reshape(obs, (obs.shape[0], -1)) # flatten it
            obs = nnx.relu(self.reshaper_linear(obs))
            
                            
        eig_vec = self.eig_linear(obs)
                
        return eig_vec
from flax import nnx
import jax.numpy as jnp

class DR_Encoder(nnx.Module):

    def __init__(self, obs_dim: int, feat_dim: int, obs_type: str, rngs: nnx.Rngs):

        self.obs_type = obs_type

        if self.obs_type == 'image':

            obs_dim = feat_dim

            self.eig_conv = nnx.Sequential(*[
                nnx.Conv(1, 16, kernel_size=(8, 8), strides = 4, rngs=rngs),
                nnx.relu,
                nnx.Conv(16, 16, kernel_size=(4, 4), strides = 2, rngs=rngs),
                nnx.relu,
                nnx.Conv(16, 16, kernel_size=(3, 3), strides = 2, rngs=rngs),
                nnx.relu,
                nnx.Flatten()
            ])
            
            self.reshaper_linear = nnx.Linear(144, obs_dim, rngs=rngs)

        self.eig_linear = nnx.Sequential(*[
            nnx.Linear(obs_dim, 1, rngs=rngs),
            # nnx.relu,
            # nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            # nnx.relu,
            # nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            # nnx.relu,
            # nnx.Linear(feat_dim, 1, rngs=rngs),
        ])

    def __call__(self, obs):

        if self.obs_type == 'image':
            obs = self.eig_conv(obs)
            obs = nnx.relu(self.reshaper_linear(obs))
                
        eig_vec = self.eig_linear(obs)
                
        return eig_vec





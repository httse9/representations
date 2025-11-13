from minigrid_basics.function_approximation.eigenlearner_FA.eigenlearner import EigenLearner
from jax import jit, lax
from flint import arb_mat, ctx
ctx.dps = 100   # important
import numpy as np
import jax.numpy as jnp
from functools import partial
from flax import nnx

class WGLNALearner(EigenLearner):

    """
    No Anchor learning version of WGL
    !!! In eigenlearning_fa.py, need to load dataset without "_2"
    """

    def __init__(self, env, dataset, test_set, args):
        super().__init__(env, dataset, test_set, args)


    def compute_matrix(self):
        WGL = self.R_inv_sqrt @ (np.eye(self.env.num_states) - self.P) @ self.R_inv_sqrt
        self.matrix = WGL

    def compute_top_eigvec(self):
        """
        Compute top log eigenvector of DR
        """
        matrix = arb_mat(self.matrix.tolist())
        lamb, e = matrix.eig(right=True, algorithm="approx")
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = lamb.argsort()
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[1]

        if e0.sum() > 0:
            e0 *= -1

        self.true_eigvec = e0 

    def eigvec(self):
        eigvec = compute_eigvec(self.encoder, self.test_set, self.env.terminal_idx[0])
        if eigvec.sum() > 0:    # flip to a direction consistent with true eigvec
            eigvec *= -1
        return eigvec


    def update(self):
        obs, actions, rewards, next_obs, next_rewards, terminals = self.processed_dataset
        step(self.encoder, self.optimizer, obs, rewards, next_obs, next_rewards, terminals)


@partial(nnx.jit, static_argnums=())
def compute_eigvec(encoder, test_set, terminal_idx):
    eigvec = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
    return eigvec


@partial(nnx.jit, static_argnums=())
def step(encoder, optimizer, obs, rewards, next_obs, next_rewards, terminals):

    def loss_fn(encoder, obs, rewards, next_obs, next_rewards, terminals):

        rewards = jnp.expand_dims(rewards, 1)
        next_rewards = jnp.expand_dims(next_rewards, 1)
        terminals = jnp.expand_dims(terminals, 1)

        phi = encoder(obs)
        next_phi = encoder(next_obs) 

        loss = (lax.stop_gradient(jnp.exp(rewards) * phi - jnp.exp((rewards + next_rewards) / 2) * next_phi) * phi).mean()

        # normalization
        loss += ((phi ** 2).mean() - 1) ** 2

        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(encoder, obs, rewards, next_obs, next_rewards, terminals)
    optimizer.update(grads)
    return loss

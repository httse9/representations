from minigrid_basics.function_approximation.eigenlearner_FA.eigenlearner import EigenLearner
from jax import jit, lax
from flint import arb_mat, ctx
ctx.dps = 100   # important
import numpy as np
import jax.numpy as jnp
from functools import partial
from flax import nnx

class WTLearner(EigenLearner):

    def __init__(self, env, dataset, test_set, args):
        super().__init__(env, dataset, test_set, args)


    def compute_matrix(self):
        WT = - self.R_inv_sqrt @ self.P @ self.R_inv_sqrt
        self.matrix = WT

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
        e0 = e[0]

        if e0.sum() < 0:
            e0 *= -1

        self.true_eigvec = np.log(e0) 

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
    return eigvec.at[terminal_idx].set(0)


@partial(nnx.jit, static_argnums=())
def step(encoder, optimizer, obs, rewards, next_obs, next_rewards, terminals):

    def loss_fn(encoder, obs, rewards, next_obs, next_rewards, terminals):

        rewards = jnp.expand_dims(rewards, 1)
        next_rewards = jnp.expand_dims(next_rewards, 1)
        terminals = jnp.expand_dims(terminals, 1)

        phi = encoder(obs)
        next_phi = encoder(next_obs) * (1 - terminals)  # if next state terminal, set to 0

       

        # normalization
        loss += ((phi ** 2).mean() - 1) ** 2

        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(encoder, obs, rewards, next_obs, next_rewards, terminals)
    optimizer.update(grads)
    return loss

"""
Onehot
- dayan: done
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --env dayan --step_size_end 1e-4 --n_epochs 30000
- fourooms: done    
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --env fourrooms --step_size_end 1e-4 --n_epochs 30000
- gridroom: done
  - rmsprop, need small step size (1e-4) large don't work.
  - switch to adam optimizer. converges much faster
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --env gridroom --step_size_end 1e-4 --step_size_start 1e-4 --n_epochs 50000

- dayan_2: done (lambd=10)
  - use lambd = 10. When lambd = 1, ground-truth eigvec is almost constant for all non-low-reward states, and for all low-reward states
  - converge rate is also much faster. Probably helps with reward propagating thru the space
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --env dayan_2 --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 10
- fourrooms_2: done (lambd=10)
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --env fourrooms_2 --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 10
- gridroom_2: 
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --env gridroom_2 --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 10


Coordinates
- dayan: done
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type coordinates --env dayan --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 10
- fourrooms:done
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type coordinates --env fourrooms --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 10
- gridroom: done
  - takes a long time, but reaches 0.995
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type coordinates --env gridroom --step_size_start 1e-4   --n_epochs 100000 --lambd 20

- dayan_2: done, easy
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type coordinates --env dayan_2 --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 20
- fourrooms_2:done
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type coordinates --env fourrooms_2 --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 10000 --lambd 20
- gridroom_2: done
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type coordinates --env gridroom_2 --step_size_start 1e-4 --step_size_end 1e-4  --n_epochs 100000 --lambd 20
  - takes a long time, might need step size decay

Image
- dayan:
  - I think works, just needs longer, reaches 0.99
  - python -m minigrid_basics.function_approximation.eigenlearning_fa --obs_type image --env dayan --step_size_start 1e-4 --step_size_end 3e-5  --n_epochs 20000 --lambd 20
  
"""

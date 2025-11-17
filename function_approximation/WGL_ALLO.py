from minigrid_basics.function_approximation.eigenlearner_FA.eigenlearner import EigenLearner
from jax import jit, lax
from flint import arb_mat, ctx
ctx.dps = 100   # important
import numpy as np
import jax.numpy as jnp
from functools import partial
from flax import nnx
from jax import random

class WGLALLOLearner(EigenLearner):

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
        e0 = e[1: 1 + self.args.eig_dim]

        e0[e0.sum(1) > 0] *= -1

        self.true_eigvec = e0.T   # columns are the eigenvectors

    def eigvec(self):
        
        eigvec = compute_eigvec(self.encoder, self.test_set, self.env.terminal_idx[0])
        
        col_sums = jnp.sum(eigvec, axis=0)           # shape (d,)
        flip_mask = jnp.where(col_sums > 0, -1.0, 1.0) 
        eigvec *= flip_mask

        return eigvec

    def update(self):
        obs, actions, rewards, next_obs, next_rewards, terminals = self.processed_dataset
        step(self.encoder, self.optimizer, obs, rewards, next_obs, next_rewards, terminals)

    def cos_sim(self, eigvec):
        return compute_cos_sim(eigvec, self.true_eigvec)
    
    def learn(self):

        eigvec = self.eigvec()
        self.norms.append((eigvec ** 2).sum(0).mean())
        self.cos_sims.append(self.cos_sim(eigvec))

        key = random.key(self.args.seed)

        update_count = 0

        for _ in range(self.args.n_epochs):

            key, subkey = random.split(key)
            batches = get_batches(self.args.batch_size, subkey, *self.processed_dataset)

            key, subkey = random.split(key)
            batches_2 = get_batches(self.args.batch_size, subkey, self.processed_dataset[0])

            # loop over minibatches
            for (s, a, r, ns, nr, t), s2 in zip(batches, batches_2):
                
                s2 = s2[0]  # unwrap tupple

                step(self.encoder, self.optimizer, s, r, ns, nr, t, s2)
                update_count += 1

                
                if update_count % self.args.log_interval == 0:
                    eigvec = self.eigvec()
                    self.norms.append((eigvec ** 2).sum(0).mean())
                    self.cos_sims.append(self.cos_sim(eigvec))

                    print(self.cos_sims[-1])
                

            
@partial(nnx.jit, static_argnums=())
def compute_cos_sim(eigvec, true_eigvec):
    eigvec_norm = jnp.linalg.norm(eigvec, axis=0)
    true_eigvec_norm = jnp.linalg.norm(true_eigvec, axis=0)
    return jnp.abs((eigvec * true_eigvec).sum(0)) / (eigvec_norm * true_eigvec_norm + 1e-8) # 1e-8 for numerical stability


@partial(nnx.jit, static_argnums=())
def compute_eigvec(encoder, test_set, terminal_idx):
    eigvec = lax.stop_gradient(encoder(test_set))
    return eigvec.at[terminal_idx].set(0)


@partial(nnx.jit, static_argnums=())
def step(encoder, optimizer, obs, rewards, next_obs, next_rewards, terminals, obs_2):

    def loss_fn(encoder, obs, rewards, next_obs, next_rewards, terminals, obs_2):

        rewards = jnp.expand_dims(rewards, 1)
        next_rewards = jnp.expand_dims(next_rewards, 1)
        terminals = jnp.expand_dims(terminals, 1)

        phi = encoder(obs)
        next_phi = encoder(next_obs) * (1 - terminals) # if next state terminal, set to 0
        phi_2 = encoder(obs_2)

        graph_loss = 0.5 * ((jnp.exp(rewards / 2) * phi - jnp.exp(next_rewards / 2) * next_phi) ** 2).mean(0).sum()

        barrier_coefficients = encoder.barrier_coefs
        duals = encoder.duals
        n = phi.shape[0]

        inner_product_matrix_1 = jnp.einsum(
            'ij,ik->jk', phi, lax.stop_gradient(phi)
        ) / n
        inner_product_matrix_2 = jnp.einsum(
            'ij,ik->jk', phi_2, lax.stop_gradient(phi_2)
        ) / n

        error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(encoder.eig_dim))
        error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(encoder.eig_dim))

        error_matrix = (error_matrix_1 + error_matrix_2) / 2

        dual_loss_pos = (lax.stop_gradient(jnp.asarray(duals)) * error_matrix).sum()
        
        dual_loss_neg = - (duals * lax.stop_gradient(error_matrix)).sum()

        quadratic_error_matrix = error_matrix_1 * error_matrix_2
        barrier_loss_pos = barrier_coefficients * quadratic_error_matrix.sum()

        allo_loss = dual_loss_pos + barrier_loss_pos + dual_loss_neg + graph_loss

        return allo_loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(encoder, obs, rewards, next_obs, next_rewards, terminals, obs_2)
    optimizer.update(grads)
    return loss


def get_batches(batch_size, key, *datasets):

    N = datasets[0].shape[0]
    idx = random.permutation(key, N)

    for start in range(0, N, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield tuple(ds[batch_idx] for ds in datasets)
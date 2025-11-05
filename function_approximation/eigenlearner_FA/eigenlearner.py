from minigrid_basics.function_approximation.encoder import DR_Encoder
from flax import nnx
import optax
import jax
from jax import jit, lax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

class EigenLearner:

    def __init__(self, env, dataset, test_set, args, lambd=1.0):
        self.env = env
        self.dataset = dataset
        self.test_set = test_set
        self.args = args
        self.lambd = lambd

        ### useful objects to have
        r = self.env.rewards
        p = self.env.transition_probs

        # uniform random policy
        pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions
        self.P = (p * pi[..., None]).sum(1)       # P as defined in the DR paper.
        self.R = np.diag(np.exp(-r / lambd))             # R as defined in the DR paper.
        self.R_inv = np.diag(np.exp(r / lambd))         # R^(-1)
        self.R_inv_sqrt = np.diag(np.exp(r / lambd / 2))         # R^(-1/2)

    def init_learn(self):
        """
        1. initialize neural net
        2. initialize optimizer
        3. compute ground-truth eigenvector for cos sim
        """
        self.init_encoder()
        self.init_optimizer()
        self.process_dataset()


        self.compute_matrix()
        self.compute_top_eigvec()

        self.cos_sims = []
        self.norms = []

    def init_encoder(self):

        if self.args.obs_type == "onehot":
            obs_dim = self.env.num_states
            feat_dim = 256
        elif self.args.obs_type == "coordinates":
            obs_dim = 2
            feat_dim = 256
        elif self.args.obs_type == "image":
            obs_dim = 256
            feat_dim = 256

        rngs = nnx.Rngs(self.args.seed)

        self.encoder = DR_Encoder(obs_dim, feat_dim, 1, 0, 0.5, self.args.obs_type, rngs)

    def init_optimizer(self):
        step_size_schedule = optax.linear_schedule(
            init_value=self.args.step_size_start,
            end_value=self.args.step_size_end,
            transition_steps=self.args.n_epochs
        )

        self.optimizer = nnx.Optimizer(
            self.encoder,
            optax.chain(
                optax.clip_by_global_norm(self.args.grad_norm_clip),
                optax.adam(step_size_schedule)
            )
        )

    def process_dataset(self):
        self.processed_dataset = [jnp.array(x) for x in zip(*self.dataset)]

    def compute_matrix(self):
        """
        Compute each matrix        
        """
        pass
    
    def compute_top_eigvec(self):
        """
        Matrix-specific computation of ground-truth eigvec
        """
        self.true_eigvec = None


    def eigvec(self):
        return compute_eigvec(self.encoder, self.test_set)


    def cos_sim(self, eigvec):
        """
        Compute cosine similarity between learned eigenvec and ground-truth eigenvec
        """
        return compute_cos_sim(eigvec, self.true_eigvec)



    def learn(self):
        for i in range(self.args.n_epochs):

            self.update()

            if (i + 1) % 100 == 0:
                eigvec = self.eigvec()
                self.norms.append((eigvec ** 2).sum())
                self.cos_sims.append(self.cos_sim(eigvec))

                print(self.cos_sims[-1])



    def update(self):
        pass

@partial(nnx.jit, static_argnums=())
def compute_eigvec(encoder, test_set):
    return lax.stop_gradient(jnp.squeeze(encoder(test_set)))

@partial(nnx.jit, static_argnums=())
def compute_cos_sim(eigvec, true_eigvec):
    return eigvec @ true_eigvec / jnp.linalg.norm(eigvec) / jnp.linalg.norm(true_eigvec)
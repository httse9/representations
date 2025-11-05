import numpy as np
from minigrid_basics.function_approximation.eigenlearner.eigenlearner import EigenLearner

from flint import arb_mat, ctx
ctx.dps = 100   # important

class AWGLLearner(EigenLearner):
    """
    Asymmetric Weighted Graph Laplacian
    R^(-1) (I - P)
    """

    def __init__(self, env, dataset, lambd=1.0):

        super().__init__(env, dataset, lambd=lambd)
        
    def init_learn(self):
        super().init_learn()

        # special initialization
        # terminal state gradient is always 0, forming a natural anchor
        # we need to initialize terminal to 0
        # other states, can random, can all negative
        # if set all positive learn negative of top eigvec (just sign flip)

        # self.eigvec = np.random.normal(size=(self.env.num_states,)) #
        self.eigvec = np.zeros((self.env.num_states)) - 1
        self.eigvec[self.env.terminal_idx[0]] = 0

    def compute_matrix(self):
        AWGL = self.R_inv @ (np.eye(self.env.num_states) - self.P)
        self.matrix = AWGL
    
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
        print(lamb)
        e = e.T[idx]
        e0 = e[1]

        if e0[0] > 0:
            e0 *= -1

        self.true_eigvec = e0

    def update(self, step_size=0.001):
        """
        Matrix-specifc update to self.eigvec
        """
        gradient = np.zeros_like(self.eigvec)
        for (s, a, r, s_next, r_next, terminated) in self.dataset:

            r /= self.lambd
            r_next /= self.lambd

            gradient[s] += 2 * np.exp(r) * self.eigvec[s] - (np.exp(r) + np.exp(r_next)) * self.eigvec[s_next]

        # mean gradient
        gradient /= self.dataset_size

        # gradient += np.exp(2 * self.eigvec).sum() - 1

        # gradient = np.clip(gradient, -10, 10)

        self.eigvec -= step_size * gradient
        self.eigvec /= np.linalg.norm(self.eigvec)
        assert np.isclose(np.linalg.norm(self.eigvec), 1), np.linalg.norm(self.eigvec) 

"""
Fail to work for _2 environments
Eigengap is much smaller than without low reward region. That's the reason?

From fourrooms_2, it does seem like the eigvec we end up learning 
is a mix of the smallest few eigenvectors...

DR eigengap is quite large.
"""
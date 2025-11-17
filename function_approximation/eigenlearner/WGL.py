import numpy as np
from minigrid_basics.function_approximation.eigenlearner.eigenlearner import EigenLearner

from flint import arb_mat, ctx
ctx.dps = 100   # important

class WGLLearner(EigenLearner):
    """
    (Symmetric) Weighted Graph Laplacian
    R^(-1/2) (I - P) R^(-1/2)
    """

    def __init__(self, env, dataset, lambd=1.0):

        super().__init__(env, dataset, lambd=lambd)


        """
        TODO: Test different formulation of M matrix
        - M is diagonal. M^{-1} should be non-negative
        - M must be positive.
        - Higher rewards should correspond to smaller value in M
        0) M = exp(diag(-r)) ensures between 0 and 1
        1) M = diag(-r), dividing by a constant does not affect eigenvectors
        """
        
        ### M = diag(-r)
        # r = self.env.rewards 
        # self.R = np.diag((-r) ** (1/self.lambd))
        # self.R_inv = np.linalg.inv(self.R)
        # self.R_inv_sqrt = np.sqrt(self.R_inv)


        
        
    def init_learn(self):
        super().init_learn()
        
        self.eigvec = np.zeros((self.env.num_states)) - 0.1
        self.eigvec[self.env.terminal_idx[0]] = 0


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

    def update(self, step_size=0.001):
        """
        Matrix-specifc update to self.eigvec
        """
        gradient = np.zeros_like(self.eigvec)
        for (s, a, r, s_next, r_next, terminated) in self.dataset:

            r /= self.lambd
            r_next /= self.lambd

            gradient[s] += np.exp(r) * self.eigvec[s] - np.exp((r + r_next) / 2) * self.eigvec[s_next]

        # mean gradient
        gradient /= self.dataset_size

        self.eigvec -= step_size * gradient

"""
lambd = 10 looks nice

python -m minigrid_basics.function_approximation.eigenlearning_tabular --env gridroom_2  
    --n_epochs 100000 --step_size 3 --lambd=20
"""
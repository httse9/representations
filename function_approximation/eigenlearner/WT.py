import numpy as np
from minigrid_basics.function_approximation.eigenlearner.eigenlearner import EigenLearner

from flint import arb_mat, ctx
ctx.dps = 100   # important

class WTLearner(EigenLearner):
    """
    (Symmetric) Weighted Transition
    - R^(-1/2) P R^(-1/2)
    """

    def __init__(self, env, dataset, lambd=1.0):
        super().__init__(env, dataset, lambd=lambd)

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

    def update(self, step_size=0.001):
        """
        Matrix-specifc update to self.eigvec
        """
        gradient = np.zeros_like(self.eigvec)
        for (s, a, r, s_next, r_next, terminated) in self.dataset:

            r /= self.lambd
            r_next /= self.lambd

            gradient[s] += -(np.exp((r + r_next) / 2)) * np.exp(self.eigvec[s_next] - self.eigvec[s])


        # mean gradient
        gradient /= self.dataset_size

        gradient += np.exp(2 * self.eigvec).sum() - 1       # normalization

        self.eigvec -= step_size * gradient


"""
Only works for large lambda (e.g., 10, 20)

NIKE JUST DO IT
python -m minigrid_basics.function_approximation.eigenlearning_tabular --env fourrooms_2 --n_epochs 10000 --step_size 1 --lambd 20
"""
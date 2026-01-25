import numpy as np
from minigrid_basics.function_approximation.eigenlearner.eigenlearner import EigenLearner

from flint import arb_mat, ctx
ctx.dps = 100   # important

class DRLearner(EigenLearner):

    def __init__(self, env, dataset, lambd=1.0):

        super().__init__(env, dataset, lambd=lambd)
        

    def compute_matrix(self):
        # DR = self.sym(np.linalg.inv(self.R - self.P))
        DR = self.R - self.sym(self.P)
        self.matrix = DR
    
    def compute_top_eigvec(self):
        """
        Compute top log eigenvector of DR
        """

        matrix = arb_mat(self.matrix.tolist())
        lamb, e = matrix.eig(right=True, algorithm="approx")
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        # lamb = np.abs(lamb)
        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        print(lamb)
        # quit()
        e = e.T[idx]
        e0 = e[0]

        if e0[0] < 0:
            e0 *= -1

        assert (e0 > 0).all()

        log_e0 = np.log(e0)


        self.true_eigvec = log_e0

    def update(self, step_size=0.001):
        """
        Matrix-specifc update to self.eigvec
        """
        # s, a, r, s_next, r_next, terminated = map(np.array, zip(*self.dataset))

        gradient = np.zeros_like(self.eigvec)
        for (s, a, r, s_next, r_next, terminated) in self.dataset:
            r /= self.lambd
            r_next /= self.lambd 

            # print(s, a, r, s_next, r_next)

            gradient[s] += np.exp(-r) - np.exp(self.eigvec[s_next] - self.eigvec[s])

        # mean gradient
        gradient /= self.dataset_size
        gradient = np.clip(gradient, -10, 10)

        self.eigvec -= step_size * gradient

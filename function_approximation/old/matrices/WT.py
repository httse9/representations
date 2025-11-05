import numpy as np
import os
import gym
import matplotlib.pyplot as plt
from flint import arb_mat, ctx

# testing imports
from minigrid_basics.examples.rep_utils import *
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ctx.dps = 100   # important
# print(ctx)

def sym(M):
    return (M + M.T) / 2


directions =np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])

class WT:
    """
    Weighted transition matrix
    R^{-1} P
    """


    def __init__(self, env):
        """
        env: environment
        gamma: discount factor for SR
        lamb: lambda for DR and MER
        """
        self.env = env


    def matrix(self, pi = None, lambd=1.0):
        """
        Compute DR with respect to pi.
        If pi is None, use uniform random policy.
        """
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        r = self.env.rewards
        P = self.env.transition_probs

        R_inv = np.diag(np.exp(r / lambd))

        P_pi = (P * pi[..., None]).sum(1)
        
        WT = R_inv @ P_pi

        return WT



    
    def eigenvector(self, pi=None, lambd=1.0, normalize=True, symmetrize=False):
        """
        Compute the log of largest eigenvector of WT
        """
        WT = self.matrix(pi=pi, lambd=lambd)
        # if symmetrize:
        #     WT = sym(WT)

        WT = arb_mat(WT.tolist())
        lamb, e = WT.eig(right=True, algorithm="approx", )
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[0]

        if e0.sum() < 0:
            e0 *= -1
        
        e0[e0 == 0] = e0[e0 != 0].min()
        assert (e0 > 0).all(), e0

        log_e0 = np.log(e0)

        if normalize:
            log_e0 /= np.sqrt(log_e0 @ log_e0)
            assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0

    
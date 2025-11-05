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

class DR:
    """
    Takes environment, compute 
    1. successor representation
    2. default representation
    3. maximum entropy representation

    4. shaping reward for SR
    5. shaping reward for DR (diff variants)
    6. shaping reward for MER (diff variants)
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

        R = self.env.rewards
        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)
        DR = np.linalg.inv(np.diag(np.exp(-R / lambd)) - P_pi)

        return DR.copy()

    def matrix_2(self, pi = None, lambd=1.0):
        """
        Compute DR with respect to pi.
        If pi is None, use uniform random policy.
        """
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        R = self.env.rewards
        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)
        P_pi = sym(P_pi)
        DR = np.linalg.inv(np.diag(np.exp(-R / lambd)) - P_pi)

        return DR.copy()


    
    def eigenvector(self, pi=None, lambd=1.0, normalize=True, symmetrize=True):
        """
        Compute log of the top eigenvector of the DR.
        """
        DR = self.matrix(pi=pi, lambd=lambd)
        if symmetrize:
            DR = sym(DR)
        assert (DR >= 0).all()

        DR = arb_mat(DR.tolist())
        lamb, e = DR.eig(right=True, algorithm="approx", )
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[0]

        # assert all entries are positive before taking log
        if e0[0] < 0:
            e0 *= -1
        assert (e0 > 0).all()

        log_e0 = np.log(e0)

        if normalize:
            log_e0 /= np.sqrt(log_e0 @ log_e0)
            assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0

    
    def eigenvector_2(self, pi=None, lambd=1.0, normalize=True, symmetrize=True):
        """
        Compute log of the top eigenvector of the DR.
        """
        DR = self.matrix_2(pi=pi, lambd=lambd)
        assert (DR >= 0).all()

        DR = arb_mat(DR.tolist())
        lamb, e = DR.eig(right=True, algorithm="approx", )
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[0]

        # assert all entries are positive before taking log
        if e0[0] < 0:
            e0 *= -1
        assert (e0 > 0).all()

        log_e0 = np.log(e0)

        if normalize:
            log_e0 /= np.sqrt(log_e0 @ log_e0)
            assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0
    
  
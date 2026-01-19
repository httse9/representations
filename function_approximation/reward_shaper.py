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

def symmetrize(M):
    return (M + M.T) / 2


directions =np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])

class RewardShaper:
    """
    Returns
    1) top eigenvector of successor representation
    2) top eigenvector of default representation
    3) smallest eigenvector of reward-weighted Laplacian

    for reward shaping
    """


    def __init__(self, env):
        """
        env: environment
        gamma: discount factor for SR
        lamb: lambda for DR and MER
        """
        self.env = env


    def compute_SR(self, pi = None, gamma=0.99):
        """
        Compute SR with respect to pi.
        If pi is None, use uniform random policy.
        """
        
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)

        SR = np.linalg.inv(np.eye(self.env.num_states) - gamma * P_pi)

        return SR.copy()
    

    
    def SR_top_eigenvector(self, pi=None, gamma=0.99):
        # make symmetric to ensure real eigenvectors and eigenvalues
        SR = symmetrize(self.compute_SR(pi=pi, gamma=gamma))

        lamb, e = np.linalg.eig(SR)
        idx = lamb.argsort()
        e = e.T[idx[::-1]]
        e0 = np.real(e[0]) 

        # normalize
        e0 /= np.sqrt(e0 @ e0)
        assert np.isclose(e0 @ e0, 1.0)

        if e0.sum() > 0:
            e0 *= -1

        return e0
    
    
    def normalize(self, e):
        """
        Normalize 
        """
        assert len(e) == self.env.num_states

        # normalize so that mean of reward diff of neighboring states is 1.
        e /= np.mean(self.compute_neighboring_diff(e))
        assert np.isclose(np.mean(self.compute_neighboring_diff(e)), 1.0)

        return e
    
    def compute_neighboring_diff(self, v):
        """
        Compute the difference of neighboring states of the vector v.
        """
        diffs = []

        for i in range(self.env.num_states):    # enumerate all states
            pos = self.env.state_to_pos[i]

            for dir in directions:      # enumerate all neighboring states
                neighbor_pos = pos + dir

                try:
                    j = self.env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * self.env.width]
                    if j >= 0:      # compute difference if neighbor exists
                        diffs.append(np.abs(v[i] - v[j]))
                except:
                    pass

        return diffs
    

        

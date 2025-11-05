"""
Study the different formulations that we can have for the DR.
The different formulations arise from the different approaches to address the problem
of the DR being not invertible when we have an absorbing terminal state with 0 reward.

Formulation 1 (used in NeurIPS 2025 paper)
Treat the terminal state as a state that transitions to an invisible absorbing state with prob 1. (Refer to Chp 3.4 of RL Book)
The terminal state has reward 0.

Formulation 2
Treat the terminal state as an absorbing state that transitions to itself with prob 1.
The terminal state has negative reward.
For a goal state, we need to ensure that the goal state has a reward that is higher than that of any other states.
This is necessary for the top eigenvector to point towards the goal state. 
In practice, we use a small number like -0.001. 

Note:
We have to separate the problem and the solution.
The formulation that we choose for the DR is part of the solution.
The problem has never changed, which is to find the optimal policy that takes the agent from start to goal.
The formulation used for the DR is just part of the solution, and does not need to align with the problem setting.
For example, the terminal state certainly does not transition to itself with prob 1 with non-zero reward in the problem setting,
but for the sake of computing the DR, which drives exploration and other stuff, we can formulate it as such.

"""

import numpy as np
import random
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import subprocess
import glob
from flint import arb_mat, ctx


from flax import nnx
from functools import partial
import optax
import gym
import os
from os.path import join
import pickle
from copy import deepcopy
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.function_approximation.encoder import DR_Encoder
from minigrid_basics.function_approximation.matrices.WGL import WGL
from minigrid_basics.function_approximation.matrices.WT import WT
from minigrid_basics.function_approximation.matrices.WL import WL
import matplotlib.pyplot as plt
import argparse
import gin
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def create_env(env_id, formulation, debug=True):
    formulation = int(formulation) # just in case formulation is string
    """
    For discussion on formulations, see comment at the top.
    """
    env = gym.make(env_id)
    if formulation == 1:
        env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=False)
    elif formulation == 2:
        env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True, goal_absorbing_reward=-1e-3)
        """
        Note that when we set goal_absorbing, we are not changng the underlying MDP formulation.
        We are just changing how we formulate the DR, which is part of the solution.
        The terminal state in the MDP formulation is still not absorbing, and still has reward 0.
        But we formulate it as absorbing and having a minor negative reward in the computation of the DR.
        """

    if debug:
        print("*****************")
        print(f"Formulation {formulation}")
        print(f"Terminal state reward in constructed reward vector: {env.rewards[env.terminal_idx[0]]}")
        print(f"Terminal state transition probs in constructed P: {env.transition_probs[env.terminal_idx[0], :, env.terminal_idx[0]]}")
        

        env.reset()
        x, y = env.state_to_pos[env.terminal_idx[0]]
        env.unwrapped.agent_pos = [x, y]
        ns, r, done, d = env.step(0)
        print(f"Actual reward received at terminal state: {r}")

        print("*****************")

    # verify that envs have all 0 terminal reward. (done)
    # verify that R & P depends on goal_absorbing correctly. (done)
    # verify that when interacting with environment, still receives 0 reward at terminal state. (done)
    # verify that DR (using RewardShaper) is computed using R & P (done)

    return env


def eigvec_myopic_policy(env, eigvec):
    """
    Get the myopic (hill-climbing policy) for current eigenvector
    """
    termination = np.zeros((env.num_states))
    policy = np.zeros((env.num_states))

    for s in range(env.num_states):

        # handle unvisited state / terminal state
        if s in env.terminal_idx:
            termination[s] = 1
            continue

        # for visited states:
        pos = env.state_to_pos[s]  # (x, y): x-th col, y-th row
        value = eigvec[s]  # init value
        myopic_a = -1

        for a, dir_vec in enumerate(np.array([
            [1, 0], # right
            [0, 1], # down
            [-1, 0],    # left
            [0, -1],    # up
        ])):
            neighbor_pos = pos + dir_vec
            neighbor_state = env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * env.width]
            
            # if neighbor state exists (not wall) 
            # and neighor state has been visited
            # and has higher eigenvector value
            # go to that neighbor state
            if neighbor_state >= 0 and eigvec[neighbor_state] > value:
                value = eigvec[neighbor_state]
                myopic_a = a

        if myopic_a == -1:
            # no better neighbor, terminate
            termination[s] = 1
        else:
            policy[s] = myopic_a

    myopic_policy = dict(termination=termination, policy=policy)
    return myopic_policy


def compute_eigenvectors(matrix):
    """
    Compute log of the top eigenvector of a matrix
    """

    matrix = arb_mat(matrix.tolist())
    lamb, e = matrix.eig(right=True, algorithm="approx", )
    lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
    e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

    return lamb, e
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dayan_2", type=str, help="Specify environment.")
    parser.add_argument("--seed", default=0, type=int, help="Initial random seed/key")
    args = parser.parse_args()

    # set random seed for env implemented in numpy
    set_random_seed(args.seed)

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = create_env(env_id, 2)
    shaper = RewardShaper(env)
    visualizer = Visualizer(env)



    ### study [R - Sym(P)]^{-1} vs Sym([R - P]^{-1})

    def check_symmetric(M, name):
        print(f"{name} symmetric: {np.allclose(M, M.T)}")

    def sym(M):
        return (M + M.T) / 2

    lambd = 1
    P = env.transition_probs
    r = env.rewards
    R = np.diag(np.exp(-r / lambd))
    R_inv = np.diag(np.exp(r / lambd))

    pi = np.ones((env.num_states, env.num_actions)) / env.num_actions

    P_pi = (P * pi[..., None]).sum(1)




    wl = WGL(env)

    e = wl.eigenvector(lambd=lambd)

    # e = -np.sqrt(R_inv) @ e

    print(e)

    # e += e[e != 0].min()
    # e = -np.log(e)

    visualizer.visualize_shaping_reward_2d(e, cmap="rainbow")
    plt.show()

    

    mp = eigvec_myopic_policy(env, e)
    visualizer.visualize_option_with_env_reward(mp)
    plt.show()

    quit()


    # diego = np.linalg.inv(R - P_pi)
    # diego = R_inv -  R_inv @ P_pi
    diego = R_inv @ P_pi
    diego = diego

    # print(diego.min())
    # print(diego.max())
    # plt.imshow(diego)
    # plt.show()


    eigvals, eigvecs = compute_eigenvectors(diego)

    # eig decomposition
    # eigvals, eigvecs = np.linalg.eig(diego)

    idx = eigvals.argsort()
    idx = np.flip(idx) 
    eigvals = eigvals[idx]
    print(eigvals)
    eigvecs = eigvecs.T[idx]

    for i, (l, v) in enumerate(zip(eigvals, eigvecs)):
        print(v)
        # if v.sum() < 0:
        #     v *= -1
        if i == 0:
            v[v == 0] = v[v != 0].min()
            v = np.log(np.abs(v))

        if v.sum() > 0:
            v *= -1

        visualizer.visualize_shaping_reward_2d(v, cmap="rainbow")
        plt.title(l)
        plt.show()
    




# 
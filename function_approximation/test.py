"""
Test eigenvectors of RWGL when there's no goal
"""

import numpy as np
import random
import pickle
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.examples.reward_shaper import RewardShaper
import matplotlib.pyplot as plt
import os
import subprocess
import glob
from minigrid_basics.function_approximation.eigenlearner import *
from os.path import join


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


def shaping_no_goal():
    env = "fourrooms_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True, no_goal=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    # print(env.terminal_idx[0])    # fourrooms: 78


    learner=  WGLLearner(env, [], lambd=20)
    learner.compute_matrix()
    lamb = learner.compute_top_eigvec()
    eigvec = learner.true_eigvec

    visualizer = Visualizer(env)
    for row in learner.true_eigvec[:11]:
        visualizer.visualize_shaping_reward_2d(row, ax=None, normalize=True, vmin=0, vmax=1, cmap="rainbow")
        plt.show()

    k_start = 1    # 0 or 1 is fine
    k = 1000
    eigvec = eigvec[k_start:k]
    lamb = lamb[k_start:k]
    
    print(lamb)

    transformed_r = eigvec - eigvec[:, 78:79]
    # print(shaping_r[:, 78])
    # print(shaping_r.shape)


    for power in [-1, -2, -3, -5, -10, -20]:
        # shaping_r = - (transformed_r ** 2).sum(0)
        shaping_r = -(transformed_r ** 2).T @ (lamb ** (power))
        # shaping_r = -np.abs(transformed_r).T @ (lamb ** (power))

        print(shaping_r.shape)

        visualizer.visualize_shaping_reward_2d(shaping_r, ax=None, normalize=True, vmin=0, vmax=1, cmap="rainbow")
        plt.show()

        mp = eigvec_myopic_policy(env, shaping_r)
        visualizer.visualize_option_with_env_reward(mp)
        plt.show()
        

if __name__ == "__main__":

    # shaping_no_goal()
    # quit()


    env = "fourrooms_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    # print(env.terminal_idx[0])    # fourrooms: 78


    learner=  WGLLearner(env, [], lambd=1)
    learner.compute_matrix()

    lamb = learner.compute_top_eigvec()
    eigvec = learner.true_eigvec

    visualizer = Visualizer(env)
    # for row in learner.true_eigvec[:11]:
    #     visualizer.visualize_shaping_reward_2d(row, ax=None, normalize=True, vmin=0, vmax=1, cmap="rainbow")
    #     plt.show()


    # print("dot", eigvec[0] @ eigvec[1])

    eigvec = eigvec[0]
    print(eigvec)
    print(eigvec[env.terminal_idx[0]])
    # print((eigvec <= 0).astype(float).mean())
    # quit()
    if eigvec.sum() <= 0:
        eigvec *= -1

    # eigvec = learner.R_inv_sqrt @ eigvec

    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap="rainbow")
    plt.show()

    mp = eigvec_myopic_policy(env, eigvec)
    visualizer.visualize_option_with_env_reward(mp)
    plt.show()
"""
Test whether it is possible to reach cosine similarity = 1 in the tabular case.
"""
import numpy as np
import random
import pickle
import gym
import gin
from copy import deepcopy
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.visualizer import Visualizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import os
import subprocess
import glob
from minigrid_basics.examples.reward_shaper import RewardShaper

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def compute_gradient_new(eigvec, dataset, learn_log):

    grad = np.zeros_like(eigvec)
    total_loss = 0

    for (s, a, r, ns, terminated) in dataset:

        if terminated:
            eigvec_ns = 0
        else:
            eigvec_ns = eigvec[ns]

        grad_s = np.exp(-r / lambd) - np.exp(eigvec_ns - eigvec[s])

        grad[s] += grad_s

    mean_grad = grad / len(dataset)

    return mean_grad


def compute_cosine_similarity(v1, v2):
    return v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":

    env_name = "gridroom_3"
    obs_type = "state_num"
    step_size = 3

    seed = 0
    # set random seed for env implemented in numpy
    set_random_seed(seed)

    # create env
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=seed, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    visualizer = Visualizer(env)

    lambd = np.abs(env.rewards).max()
    print("Lambda:", lambd)

    # load dataset
    with open(f"minigrid_basics/function_approximation/static_dataset/{env_name}_{obs_type}_2.pkl", "rb") as f:
        dataset = pickle.load(f)
    dataset_size = len(dataset)

    # compute ground-truth eigenvector
    shaper = RewardShaper(env)
    true_log_eigvec_DR = np.array(shaper.DR_top_log_eigenvector(lambd=lambd, normalize=False, symmetrize=True))

    # init eigvec
    eigvec = np.zeros((env.num_states), dtype=np.float64)

    css = []
    for e in range(200000):

        # update
        grad = compute_gradient_new(eigvec, dataset, True)
        
        eigvec -= step_size * grad 


        # Normalize log of eigenvector so that the original eigenvector has norm of 1
        # !!! Combined with natural gradient, can get cosine sim 0.999
        # norm = np.linalg.norm(np.exp(eigvec))
        # eigvec -= np.log(norm)

        # !!! Cosine similarity around 3.83, but retrieves true eigenvector visually
        # Low cosine similarity probably because of adding a constant does not preserve cosine
        # similarity.
        # eigvec -= eigvec.mean()
        # print(np.mean(eigvec))

        # anchor
        # !!! Gets cosine similarity 0.998
        # eigvec[env.terminal_idx[0]] = 0

        # anchor 2
        # eigvec -= eigvec[env.terminal_idx[0]]
        
        cs = compute_cosine_similarity(eigvec, true_log_eigvec_DR)
        css.append(cs)
        print(cs)


    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)

    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(true_log_eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1)

    plt.show()

    plt.plot(css)
    plt.show()


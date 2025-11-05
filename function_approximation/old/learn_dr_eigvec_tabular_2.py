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

def compute_gradient_new(eigvec, dataset):

    grad = np.zeros_like(eigvec)
    total_loss = 0

    for (s, a, r, ns, terminated) in dataset:

        if terminated:
            eigvec_ns = 0

            grad[ns] += 2       
            # exp(0) is 1. But for this dataset, terminal state is not directly sampled, and only
            # appears when a neighbor transitions to the terminal state. To account for the diff, need 2, so that total is 4.
        else:
            eigvec_ns = eigvec[ns]

        grad_s = np.exp(-r / lambd) - np.exp(eigvec_ns - eigvec[s])

        grad[s] += grad_s

    mean_grad = grad / len(dataset)
    return mean_grad

def compute_gradient(eigvec, dataset):
    """
    For dataset without a '_2'
    terminated means whether s is terminal state here
    """
    grad = np.zeros_like(eigvec)
    total_loss = 0

    for (s, a, r, ns, terminated) in dataset:

        if not terminated:
            grad_s = np.exp(-r / lambd) - np.exp(eigvec[ns] - eigvec[s])

        else:
            # if terminal state, no transition to next state, so second term vanishes 
            grad_s = np.exp(-r / lambd)
            
        grad[s] += grad_s

    mean_grad =  grad / len(dataset)  # mean over all transitions

    mean_grad += np.exp(2 * eigvec).mean() - 1

    return mean_grad

def compute_gradient_per_transition(eigvec, transition):
    s, a, r, ns, terminated = transition

    if terminated:
        eigvec_ns = 0
    else:
        eigvec_ns = eigvec[ns]

    grad_s = np.exp(-r / lambd) - np.exp(eigvec_ns - eigvec[s])

    return s, grad_s


def compute_cosine_similarity(v1, v2):
    return v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":

    env_name = "gridroom_2"
    obs_type = "state_num"
    step_size = 0.3

    update_each_transition = False

    learning_mode = 1

    seed = 0
    # set random seed for env implemented in numpy
    set_random_seed(seed)

    # create env
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=seed, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=False)      # use old formulation

    visualizer = Visualizer(env)
    shaper = RewardShaper(env)

    lambd = 1.3 #np.abs(env.rewards).max()
    print("Lambda:", lambd)

    # load dataset
    if learning_mode == 1:
        dataset_file = f"minigrid_basics/function_approximation/static_dataset/{env_name}_{obs_type}.pkl"
    else: 
        dataset_file = f"minigrid_basics/function_approximation/static_dataset/{env_name}_{obs_type}_2.pkl"

    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)
    dataset_size = len(dataset)

    # compute ground-truth eigenvector
    true_log_eigvec_DR = np.array(shaper.DR_top_log_eigenvector(lambd=lambd, normalize=False, symmetrize=True))

    lambd = 1   # simply use lambd=1 when learning eigenvector directly

    ### init eigvec
    eigvec = np.zeros((env.num_states), dtype=np.float64)   # all zeros
    ## some ways of randomly initializing, to test robustness
    # eigvec = np.random.uniform(low=-1.0, high=1.0, size=env.num_states).astype(np.float64)
    # eigvec = np.random.normal(scale=5, size=env.num_states).astype(np.float64)


    css = []
    for e in range(100000):

        if update_each_transition:
            
            for transition in dataset:
                s, grad_s = compute_gradient_per_transition(eigvec, transition)

                # clipping is necessary, otherwise "inf" very easily
                grad_s = np.clip(grad_s, -2, 2)

                eigvec[s] -= step_size * grad_s


        else: # batch update

            if learning_mode == 1:
                grad = compute_gradient(eigvec, dataset)       
            elif learning_mode == 2:
                grad = compute_gradient_new(eigvec, dataset)   


            # clipping
            grad = np.clip(grad, -10, 10)

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

    print(eigvec)


    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)

    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(true_log_eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1)

    plt.show()

    plt.plot(css)
    plt.show()


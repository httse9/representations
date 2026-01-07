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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(env_name):
    with open(f"minigrid_basics/function_approximation/static_dataset/{env_name}_state_num.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset


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

def eigenlearning_tabular(args, env):
    
    dataset = []# load_dataset(args.env)
    visualizer = Visualizer(env)

    # visualizer.visualize_env()
    # plot_dir = "minigrid_basics/function_approximation/plots/choosing_M"
    # os.makedirs(plot_dir, exist_ok=True)
    # plt.savefig(join(plot_dir, "env.png"), dpi=300)
    # plt.show()


    learner = DRLearner(env, dataset, lambd=args.lambd)
    # learner = AWGLLearner(env, dataset, lambd=args.lambd)
    # learner = WGLLearner(env, dataset, lambd=args.lambd)
    # learner = AWTLearner(env, dataset, lambd=args.lambd)
    # learner = WTLearner(env, dataset, lambd=args.lambd)
    learner.init_learn()

    # lamb = learner.compute_top_eigvec()
    # print(lamb)

    # for e in learner.true_eigvec:
    #     visualizer.visualize_shaping_reward_2d(e, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    #     plt.show()

    # eigvec = learner.true_eigvec[1:]
    # eigvec = eigvec * (1 / np.sqrt(lamb[1:])).reshape(-1, 1)
    # eigvec = eigvec - eigvec[:, env.terminal_idx[0]: env.terminal_idx[0] + 1]
    # eigvec = -(eigvec ** 2).sum(0)

    eigvec = learner.true_eigvec
    # eigvec /= np.linalg.norm(eigvec)
    print(eigvec, eigvec[env.terminal_idx[0]])
    print(np.exp(eigvec), np.exp(eigvec)[env.terminal_idx[0]])
    # plt.plot(eigvec)
    # eigvec = learner.R_inv_sqrt @ eigvec
    # eigvec /= np.linalg.norm(eigvec)
    # print(eigvec)
    # plt.plot(eigvec)
    # plt.show()

    # learner = AWGLLearner(env, dataset, lambd = args.lambd)
    # learner.init_learn()
    # print(learner.true_eigvec)
    # print(eigvec @ learner.true_eigvec / (np.linalg.norm(eigvec) * np.linalg.norm(learner.true_eigvec)))

    # plt.clf()
    # plt.hist(learner.true_eigvec)
    # plt.show()

    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    # plt.savefig(join(plot_dir, "increasing.png"), dpi=300)
    plt.show()


    mp = eigvec_myopic_policy(env, eigvec)
    visualizer.visualize_option_with_env_reward(mp)
    plt.show()

    quit()

    learner.learn(n_epochs=args.n_epochs, step_size=args.step_size)

    print(learner.eigvec)

    plt.plot(learner.cos_sims)
    plt.show()

    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(learner.true_eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.title("True")
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(learner.eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.title("Learned")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--lambd", help="lambda for DR", default=1.0, type=float)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--step_size", default=1e-1, type=float, help="Starting step size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # create env
    set_random_seed(args.seed)
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True,goal_absorbing_reward=-0.001)

    # learn
    cmap = "rainbow"
    eigenlearning_tabular(args, env)
import numpy as np
import jax.numpy as jnp
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
from minigrid_basics.function_approximation.eigenlearner_FA import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(args):
    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_2.pkl", "rb") as f:
        dataset = pickle.load(f)

    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_test.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f))

    return dataset, test_set

def eigenlearning_tabular(args, env):
    
    dataset, test_set = load_dataset(args)
    visualizer = Visualizer(env)

    learner = WGLLearner(env, dataset, test_set, args)
    learner.init_learn()

    print("Before", learner.eigvec()[env.terminal_idx[0]])

    learner.learn()
    

    # print(learner.true_eigvec)
    # visualizer.visualize_shaping_reward_2d(learner.true_eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    # plt.show()
    # quit()

    print(learner.eigvec())

    print("After", learner.eigvec()[env.terminal_idx[0]])


    plt.subplot(2, 1, 1)
    plt.plot(learner.cos_sims)
    plt.title("cos sim")

    plt.subplot(2, 1, 2)
    plt.plot(learner.norms)
    plt.title("Eigvec Norm")
    plt.tight_layout()
    plt.show()

    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(learner.true_eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.title("True")
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(learner.eigvec(), ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.title("Learned")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--obs_type", default="onehot", type=str)
    parser.add_argument("--lambd", help="lambda for DR", default=1.0, type=float)
    parser.add_argument("--n_epochs", type=int, default=10000)

    parser.add_argument("--step_size_start", default=1e-3, type=float, help="Starting step size")
    parser.add_argument("--step_size_end", default=1e-3, type=float, help="Ending step size")

    parser.add_argument("--grad_norm_clip", default=0.5, type=float, help="Ending step size")


    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # create env
    set_random_seed(args.seed)
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    # learn
    cmap = "rainbow"
    eigenlearning_tabular(args, env)
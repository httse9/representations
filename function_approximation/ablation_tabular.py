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
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_batches(batch_size, key, *datasets):

    N = datasets[0].shape[0]
    idx = random.permutation(key, N)

    for start in range(0, N, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield tuple(ds[batch_idx] for ds in datasets)


def load_static_dataset(args):
    if args.dr_mode == "dr_anchor":
        with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_state_num_2.pkl", "rb") as f:
            dataset = pickle.load(f)
    elif args.dr_mode == "dr_norm":
        with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_state_num.pkl", "rb") as f:
            dataset = pickle.load(f)

    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_state_num_test.pkl", "rb") as f:
        test_set = np.array(pickle.load(f))

    return dataset, test_set

def update_l_GDO(v, s, r, ns, t, b):
    
    grad = v[s] * np.exp(-r) - v[ns] + b * ((v ** 2).mean() - 1) * v[s]
    v[s] -= grad
    


def eigenlearning_tabular(args, env, ):
    
    dataset, _ = load_static_dataset(args)
    n = len(dataset)
    obs, actions, rewards, next_obs, next_rewards, terminals = [np.array(x) for x in zip(*dataset)]
    rewards /= args.lambd
    next_rewards /= args.lambd

    v = np.ones(env.num_states)

    

    for e in tqdm(range(20000)):
        grad = np.zeros(env.num_states)
        for (s, a, r, ns, nr, t) in dataset:
            grad_s = np.exp(-r / args.lambd) * v[s] - v[ns] + 0.5 * ((v**2).sum() - 1) * v[s]

            v[s] -= 0.01 * grad_s

            # grad[s] += grad_s

        # grad /= n

        # v -= grad

        # update_l_GDO(v, obs, rewards, next_obs, terminals, 0.5)

    print(v)

    visualizer.visualize_shaping_reward_2d(np.log(v), ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.show()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--lambd", help="lambda for DR", default=20.0, type=float)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--step_size", default=1e-1, type=float, help="Starting step size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dr_mode", type=str, default="dr_anchor")
    args = parser.parse_args()

    # create env
    set_random_seed(args.seed)
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True,goal_absorbing_reward=-0.001)
    shaper = RewardShaper(env)
    visualizer = Visualizer(env)

    # learn
    cmap = "rainbow"
    eigenlearning_tabular(args, env)




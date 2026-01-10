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


def main(env):

    visualizer = Visualizer(env)
    cmap="rainbow"

    env_name = "gridmaze"
    obs_type = "coordinates"
    seed = 6

    with open(f"minigrid_basics/function_approximation/experiments_dr_real/{env_name}/{obs_type}/data/20.0-3e-05-3e-05-500-0.5-{seed}.pkl", "rb") as f:
        data = pickle.load(f)

    eigvec = np.array(data['eigvec'])
    print(eigvec)

    # eigvec[eigvec == 0] = eigvec[eigvec != 0].mean()

    # # eigvec[eigvec < 0] -= eigvec[eigvec < 0].max() + 1
    # # print(eigvec)

    # visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    # plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="gridmaze", type=str, help="Specify environment.")
    parser.add_argument("--lambd", help="lambda for DR", default=1.0, type=float)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--step_size", default=1e-1, type=float, help="Starting step size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    # # create env
    # gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    # env_id = maxent_mon_minigrid.register_environment()
    # env = gym.make(env_id, disable_env_checker=True)
    # env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True,goal_absorbing_reward=-0.001)

    main([])
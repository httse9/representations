"""
2x2 plot to put in one column of the icml paper

sr row, sr eigenvector
dr row, dr eigenvector
"""


import numpy as np
import pickle
from os.path import join
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter
from copy import deepcopy
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.reward_envs import maxent_mon_minigrid
import gin
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.visualizer import Visualizer
import os
import gym

def figure():

    # get old shaper? to compute eigenvector of SR and DR
    env = "fourrooms_2"
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env_SR = gym.make(env_id, disable_env_checker=True)
    env_SR = maxent_mdp_wrapper.MDPWrapper(env_SR, goal_absorbing=False)
    shaper_SR = RewardShaper(env_SR)
    SR = shaper_SR.compute_SR(gamma=0.999)
    eigvec_SR = shaper_SR.SR_top_eigenvector(gamma=0.999)


    env_DR = gym.make(env_id, disable_env_checker=True)
    env_DR = maxent_mdp_wrapper.MDPWrapper(env_DR, goal_absorbing=True, goal_absorbing_reward=-0.001)
    shaper_DR = RewardShaper(env_DR)
    DR = shaper_DR.compute_DR()
    eigvec_DR = shaper_DR.DR_top_log_eigenvector()

    visualizer = Visualizer(env_SR)
    cmap = "rainbow"
    state_num = 31

    plt.figure(figsize=(4, 2))
    # plt.subplot(2, 2, 1)
    # visualizer.visualize_shaping_reward_2d(SR[state_num], ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(eigvec_SR, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    # plt.subplot(2, 2, 3)
    # visualizer.visualize_shaping_reward_2d(np.log(DR[state_num]), ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)

    plt.tight_layout()
    # plt.savefig("minigrid_basics/function_approximation/plots/dr_vs_sr.png", dpi=300)
    plt.show()







if __name__ == "__main__":
    figure()

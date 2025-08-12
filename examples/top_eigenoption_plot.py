import gin
import os
import gym
import matplotlib.pyplot as plt
import numpy as np
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.examples.rep_utils import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def figure_2(envs, lambds):
    """
    Visualize eigenvectors of SR, DR
    """
    plt.figure(figsize=(12, 6))


    # plot SR
    for i, (env_name, lambd) in enumerate(zip(envs, lambds)):
        gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
        env_id = maxent_mon_minigrid.register_environment()
        env = gym.make(env_id, seed=42)
        env = maxent_mdp_wrapper.MDPWrapper(env, )

        visualizer = Visualizer(env)
        shaper = RewardShaper(env)


        eigvec_SR = shaper.SR_top_eigenvector()
        if eigvec_SR.sum() > 0:
            eigvec_SR *= -1
        option_SR = compute_eigenoption(env, eigvec_SR)
        
        eigvec_DR = shaper.DR_top_log_eigenvector(lambd=lambd)
        option_DR = compute_eigenoption(env, eigvec_DR)


        plt.subplot(2, 4, i + 1)
        visualizer.visualize_option_with_env_reward(option_SR)
        # plot_value_pred_map(env, eigvec_SR, contain_goal_value=True)

        # plt.subplot(2, 4, i + 2)
        # visualizer.visualize_option(option_SR)


        plt.subplot(2, 4, i + 1 + 4)
        visualizer.visualize_option_with_env_reward(option_DR)
        # plot_value_pred_map(env, eigvec_DR, contain_goal_value=True)

        # plt.subplot(2, 4, i + 1 + 5)
        # visualizer.visualize_option(option_DR)



    plt.tight_layout()
    plt.savefig("minigrid_basics/plots/Figure_top_option.png", dpi=300)
    plt.show()

def compute_eigenoption(env, eigvec, gamma=0.99):
    """
    Q-learning with batch data.
    Construct option
    """
    Q = np.zeros((env.num_states, env.num_actions))
    P = env.transition_probs    # (s, a, s')

    while True:   
        
        Q_copy = Q.copy()
        
        for s in range(env.num_states):
            r = eigvec - eigvec[s]
            # idx = P[s, 3] > 0
            # print(s, r[idx])
            # quit()
            q_max = Q.max(1)
            q_new = P[s] @ (r + gamma * q_max)
            Q[s] = q_new
            
        if np.abs(Q_copy - Q).max() < 1e-10:
            break

    print(Q[0])
            
    pi = np.argmax(Q, axis=1)
    termination_set = Q.max(axis=1) <= 0

    option = {
        'policy': pi,
        'termination': termination_set,
        'initiation': ~termination_set
    }
    return option

if __name__ == "__main__":

    envs = [
        'dayan', 
        'fourrooms', 
        'gridroom', 
        'gridmaze', 
    ]
    reward_envs = [env + "_2" for env in envs]
    lambds = [1.3] * 8

    # run below separately
    figure_2(reward_envs, lambds)


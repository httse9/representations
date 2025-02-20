import gin
import os
import gym
import matplotlib.pyplot as plt
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.examples.visualizer import Visualizer


def figure_1(envs):
    """
    Visualize all environments
    """
    plt.figure(figsize=(12, 6))

    for i, env_name in enumerate(envs):
        gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
        env_id = maxent_mon_minigrid.register_environment()
        env = gym.make(env_id, seed=42)
        env = maxent_mdp_wrapper.MDPWrapper(env, )

        visualizer = Visualizer(env)

        plt.subplot(2, 4, i + 1)
        visualizer.visualize_env()

    plt.tight_layout()
    plt.savefig("minigrid_basics/plots/Figure_1.png", dpi=300)
    plt.show()

def figure_2(envs, lambds):
    """
    Visualize rewards of DR
    """
    plt.figure(figsize=(12, 6))

    for i, (env_name, lambd) in enumerate(zip(envs, lambds)):
        gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
        env_id = maxent_mon_minigrid.register_environment()
        env = gym.make(env_id, seed=42)
        env = maxent_mdp_wrapper.MDPWrapper(env, )

        visualizer = Visualizer(env)
        shaper = RewardShaper(env)

        eigvec_DR = shaper.DR_top_log_eigenvector(lambd=lambd)
        reward_DR = shaper.shaping_reward_transform_using_terminal_state(eigvec_DR)

        plt.subplot(2, 4, i + 1)
        visualizer.visualize_shaping_reward_2d(reward_DR)

    plt.tight_layout()
    plt.savefig("minigrid_basics/plots/Figure_2.png", dpi=300)
    plt.show()

def figure_3(envs):
    """
    Visualize rewards of SR
    """
    plt.figure(figsize=(12, 3))

    for i, env_name in enumerate(envs):
        gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
        env_id = maxent_mon_minigrid.register_environment()
        env = gym.make(env_id, seed=42)
        env = maxent_mdp_wrapper.MDPWrapper(env, )

        visualizer = Visualizer(env)
        shaper = RewardShaper(env)

        eigvec_SR = shaper.SR_top_eigenvector()
        reward_SR = shaper.shaping_reward_transform_using_terminal_state(eigvec_SR)

        plt.subplot(1, 4, i + 1)
        visualizer.visualize_shaping_reward_2d(reward_SR)

    plt.tight_layout()
    plt.savefig("minigrid_basics/plots/Figure_3.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    envs = [
        'dayan', 
        'fourrooms', 
        'gridroom', 
        'gridmaze', 
    ]
    reward_envs = [env + "_2" for env in envs]

    # lambds = [
    #     1, 1, 1, 1.1, 1, 1, 1.1, 1.3
    # ]

    lambds = [1.3] * 8

    # run below separately
    # figure_1(envs + reward_envs)

    figure_2(envs + reward_envs, lambds)

    # figure_3(envs)


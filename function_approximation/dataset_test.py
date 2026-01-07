import pickle
import numpy as np
from os.path import join
from minigrid_basics.examples.visualizer import Visualizer
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import matplotlib.pyplot as plt

if __name__ == "__main__":

    env_name = "gridmaze_2"

    gin.parse_config_file(join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    path = join("minigrid_basics", "function_approximation", "dataset", f"{env_name}_dataset.pkl")

    with open(path, "rb") as f:
        dataset = pickle.load(f)
    print("Dataset Loaded")

    obs, actions, rewards, next_obs, next_rewards, terminals = [np.array(x) for x in zip(*dataset['onehot'])]
    
    state_count = obs.sum(0)

    visualizer = Visualizer(env)
    visualizer.visualize_shaping_reward_2d(state_count, ax=None, normalize=True, vmin=0, vmax=1, cmap="Reds")
    plt.savefig(f"{env_name}_count.png")


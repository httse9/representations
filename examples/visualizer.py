import numpy as np
import os
import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# testing imports
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class Visualizer:
    """
    Visualize environment, reward, and etc.
    """

    def __init__(self, env):
        """
        env: environment
        """
        self.env = env

    def visualize_env(self,):
        """
        Visualize environment
        """
        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w, 3))

        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j] *= 0.3  # gray

                elif grid[i, j] == 'l':
                    # lava
                    image[i, j, 1] = 0.2
                    image[i, j, 2] = 0  # orange

                elif grid[i, j] == 's':
                    # agent
                    image[i, j, :] = 0 # black

                elif grid[i, j] == 'g':
                    # goal
                    image[i, j, [0, 2]] = 0 # green

        plt.imshow(image)
        plt.axis('off')


    def visualize_shaping_reward_2d(self, reward):
        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w))

        # normalize reward
        reward_normalized = reward - reward.min()
        reward_normalized /= reward_normalized.max()
        assert (reward_normalized >= 0).all() and (reward_normalized <= 1).all()

        # construct the map with reward
        # let walls have value 0 for now
        state_num = 0
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j] = 0.
                    continue

                # set reward
                image[i, j] = reward_normalized[state_num]
                state_num += 1

        # use plt cmap to get colored image
        cmap = plt.get_cmap('rainbow')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        image = scalar_map.to_rgba(image)

        # draw walls again
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    image[i, j, :3] = 0.3
        

        plt.imshow(image)
        plt.axis('off')





### testing
if __name__ == "__main__":

    env_name = "gridroom_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))


    env_id = maxent_mon_minigrid.register_environment()

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"fourrooms.gin"))
    env_id = maxent_mon_minigrid.register_environment()


    env = gym.make(env_id, seed=42)
    env = maxent_mdp_wrapper.MDPWrapper(env, )


    visualizer = Visualizer(env)

    # test environment visualization
    visualizer.visualize_env()
    plt.show()

    # test reward shaping visualization
    shaper = RewardShaper(env)

    eigvec_SR = shaper.SR_top_eigenvector()
    reward_SR = shaper.shaping_reward_transform_using_terminal_state(eigvec_SR)

    visualizer.visualize_shaping_reward_2d(reward_SR)
    plt.show()

    eigenvec_DR = shaper.DR_top_log_eigenvector(lambd=1.1)
    reward_DR = shaper.shaping_reward_transform_using_terminal_state(eigenvec_DR)
    
    visualizer.visualize_shaping_reward_2d(reward_DR)
    plt.show()

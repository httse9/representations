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

action_markers = {
    0: '>',
    1: 'v',
    2: '<',
    3: '^'
}


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
                    image[i, j] = np.array((44, 62, 80)) / 255.  # gray

                elif grid[i, j] == 'l':
                    # lava
                    image[i, j] = np.array((231, 76, 60)) / 255.    # orange

                elif grid[i, j] == 's':
                    # agent
                    image[i, j] = np.array((41, 128, 185)) / 255.   # blue

                elif grid[i, j] == 'g':
                    # goal
                    image[i, j] = np.array((46, 204, 113)) / 255. # green

        plt.imshow(image)
        plt.axis('off')


    def visualize_shaping_reward_2d(self, reward, ax=None):

        if ax is None:
            ax = plt.gca()

        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w))

        # normalize reward
        reward_normalized = reward - reward.min()
        if not (reward_normalized == 0).all():
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
                    image[i, j, :3] = np.array((44, 62, 80)) / 255.
        

        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()

    def visualize_option(self, option, ax=None):

        marker_size = 9.5

        if ax is None:
            ax = plt.gca()

        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w, 3))

        # draw walls
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j, :3] = np.array((44, 62, 80)) / 255.


        ax.imshow(image)

        for s, a, in enumerate(option['policy']):

            x, y = self.env.state_to_pos[s]

            # if termination, plot terminate sign
            if option['termination'][s]:
                ax.plot([x], [y], marker='o', markersize=marker_size, color='#c0392b')
                continue

            # if not termination set, plot action
            ax.plot([x ], [y ], marker=action_markers[a], markersize=marker_size, color="#1abc9c")
            
        ax.axis('off')
        plt.tight_layout()
        





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

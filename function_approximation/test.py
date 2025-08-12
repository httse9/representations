from minigrid_basics.examples.ROD_cycle import RODCycle
import os
import numpy as np
from flint import arb_mat, ctx
from itertools import islice

ctx.dps = 100   # important

# testing imports
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import subprocess
import glob
import pickle


if __name__ == "__main__":
    

    env_name = "dayan_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    # np.random.seed(0)

    env = gym.make(env_id, seed=None, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    # print(env.terminal_idx[0])

    s = env.reset()
    env.step(0) # take a step to the right

    ### State number of constructing one-hot vector
    print("State number:", s['state'])

    ### (x, y) coordinate
    y, x = env.agent_pos
    print("(x, y) coordinates:", (x, y))

    ### Pixels (one pixel for one grid)
    image = env.custom_rgb()
    print(image.shape)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Pixels")
    plt.show()




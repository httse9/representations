import os
from absl import app
from absl import flags
import gin
import gym
from matplotlib import cm
from matplotlib import colors
import matplotlib.pylab as plt
import numpy as np
import pickle
from os.path import join
#from tqdm import tqdm
import random
from matplotlib import cm

from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.envs import maxent_mon_minigrid
from minigrid_basics.examples.rep_utils import construct_value_pred_map
from minigrid_basics.examples.rep_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'maxent_fourrooms_2', 'Environment to run.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')

flags.DEFINE_boolean('plot', False, 'When True, plot results of experiments. When False, run the experiment.')


def plot_3d(env, v):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    value_map = construct_value_pred_map(env, v, contain_goal_value=True).T
    value_map[np.isinf(value_map)] = np.median(value_map[~np.isinf(value_map)])

    # get rid of walls
    value_map = value_map[1:-1, 1:-1]

    value_map = np.rot90(value_map, k=1).T

    x, y = value_map.shape
    x, y = np.meshgrid(range(x), range(y))
    ax.plot_surface(x, y, value_map, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.title.set_text("SR")

    plt.show()

def main(argv):
    # print(argv)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    gin.parse_config_files_and_bindings(
        [os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, '{}.gin'.format(FLAGS.env))],
        bindings=FLAGS.gin_bindings,
        skip_unknown=False)
    env_id = maxent_mon_minigrid.register_environment()

    ##############################
    ### Make env
    ##############################
    # do not treat goal as absorbing
    env = gym.make(env_id, )
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    # env_name = "MiniGrid-MaxEntFourRooms-v0"
    env_name = env_id
    path = join("minigrid_basics", "experiments", "reward_shaping_fit", env_name)

    rep = "SR"
    n_episodes = 1000
    r_shaped_weight = 1.0

    for n_episodes in [100, 500, 1000]:
        reward_shaped_list = []
        M_list = []
        for seed in range(1, 21):
            with open(join(path, f"{rep}-{n_episodes}-0-{r_shaped_weight}-{1.0}-{seed}.pkl"), "rb") as f:
                data = pickle.load(f)

            # plot_3d(env, data['reward_shaped'])

            reward_shaped_list.append(data['reward_shaped'][None])
            M_list.append(data['M'][None])

        reward_shaped = np.concatenate(reward_shaped_list).mean(0)
        plot_3d(env, reward_shaped)
        # reward_shaped /= np.abs(reward_shaped[0])
        # plot_value_pred_map(env, reward_shaped, contain_goal_value=True)
        # plt.show()
        
        # M = np.concatenate(M_list).mean(0)
        # if rep == "MER":
        #     M = np.log(M)
        # plt.imshow(M)
        # plt.show()


        



    

if __name__ == '__main__':
    app.run(main)

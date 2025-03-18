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

def SR_aux_reward(env, i=0):
    """
    i: the (i + 1)-th top eigenvector. Default is top eigenvector
    """
    # terminal_idx = np.where(~env.nonterminal_idx)[0][0]
    terminal_idx = env.terminal_idx[0]

    SR = compute_SR(env)
    if not np.allclose(SR, SR.T):   # handle assymetry, avoid imaginary numbers
        SR = (SR + SR.T) / 2

    # eigendecomposition
    lamb, e = np.linalg.eig(SR)
    idx = lamb.argsort()
    e = e.T[idx[::-1]]
    e0 = np.real(e[i])  # largest eigenvector

    e0 = - np.abs(e0[terminal_idx] - e0)      # shaped reward
    e0 /= np.abs(e0).max()  # normalize

    return e0

def DR_MER_aux_reward(env, i=0, mode="MER"):
    # terminal_idx = np.where(~env.nonterminal_idx)[0][0]
    terminal_idx = env.terminal_idx[0]
    if mode == "MER":
        DR = compute_MER(env)
    else:
        DR = compute_DR(env)

    if not np.allclose(DR, DR.T): # handle asymmetry
        DR = (DR + DR.T) / 2

    lamb, e = np.linalg.eig(DR)
    idx = lamb.argsort()
    e = e.T[idx[::-1]]
    e0 = np.real(e[i])      # get i-th eigenvector

    if (e0 < 0).astype(int).sum() > len(e0) / 2:
        e0 *= -1

    # handle #6 careful interpolate after log
    directions =np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ])
    e0_copy = e0.copy()
    e0_copy[e0_copy <= 0] = 1
    e0_copy = np.log(e0_copy)
    for i in range(len(e0_copy)):
        if e0[i] <= 0:
            pos = np.array(env.state_to_pos[i])
            neighbor_values = []
            for d in directions:
                try:
                    neighbor = pos + d
                    # print("  ", neighbor)
                    j = env.pos_to_state[neighbor[0] + neighbor[1] * env.width]
                    if j >= 0 and e0[j] > 0:
                        neighbor_values.append(e0_copy[j])
                except:
                    pass
            e0_copy[i] = np.mean(neighbor_values)
    e0 = e0_copy

    e0 = - np.abs(e0[terminal_idx] - e0)      # shaped reward
    e0 /= np.abs(e0).max()  # normalize

    return e0


def plot_3d(env, v):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    value_map = construct_value_pred_map(env, v, contain_goal_value=True).T
    value_map[np.isinf(value_map)] = np.median(value_map[~np.isinf(value_map)])

    # get rid of walls
    value_map = value_map[1:-1, 1:-1]

    # value_map = np.rot90(value_map, k=3).T

    x, y = value_map.shape
    x, y = np.meshgrid(range(x), range(y))
    ax.plot_surface(x, y, value_map, cmap=cm.rainbow, linewidth=0.2, antialiased=False, edgecolors='k')
    # ax.title.set_text("SR")

    ax.set_zlim(-1, 0)

    # plt.show()

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

    path = 'minigrid_basics/plots/reward_shaping/'

    #####
    # Visualization of environment
    #####
    grid = env.raw_grid.T
    h, w = grid.shape
    image = np.ones((h, w, 3))

    for i in range(h):
        for j in range(w):
            if grid[i, j] == '*':
                # wall
                image[i, j] *= 0.5  # gray

            elif grid[i, j] == 'l':
                # lava
                image[i, j, 1] = 0.33 
                image[i, j, 2] = 0  # orange

            elif grid[i, j] == 's':
                # agent
                image[i, j, :2] = 0 # blue

            elif grid[i, j] == 'g':
                # goal
                image[i, j, [0, 2]] = 0 # green

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'minigrid_basics/plots/reward_shaping/{FLAGS.env}.png', dpi=300)
    plt.clf()

    ### Visualization of closed form SR MER
    SR =  compute_SR(env)
    MER = compute_MER(env)

    plt.imshow(SR)
    plt.axis('off')
    plt.savefig(join(path, f"{FLAGS.env}_SR.png"), dpi=300)
    plt.clf()
    
    plt.imshow(MER)
    plt.axis('off')
    plt.savefig(join(path, f"{FLAGS.env}_MER.png"), dpi=300)
    plt.close()

    ##### 
    # Visualize shaped reward
    #####

    SR_reward = SR_aux_reward(env)
    MER_reward = DR_MER_aux_reward(env, mode='MER')

    plot_3d(env, SR_reward)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(join(path, f"{FLAGS.env}_SR_reward.png"), dpi=300)
    plt.clf()


    plot_3d(env, MER_reward)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(join(path, f"{FLAGS.env}_MER_reward.png"), dpi=300)
    plt.clf()

    # R = env.rewards
    # R /= np.abs(R).max()
    # plot_3d(env, R)
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.show()


    quit()


    # env_name = "MiniGrid-MaxEntFourRooms-v0"
    env_name = env_id
    path = join("minigrid_basics", "experiments", "reward_shaping_td", env_name)

    rep = "MER"
    n_episodes = 1000
    r_shaped_weight = 1.0

    for n_episodes in [100, 500, 1000]:
        reward_shaped_list = []
        M_list = []
        for seed in range(1, 21):
            with open(join(path, f"{rep}-{n_episodes}-0-{r_shaped_weight}-{1.0}-{seed}.pkl"), "rb") as f:
                data = pickle.load(f)

            reward_shaped_list.append(data['reward_shaped'][None])
            M_list.append(data['M'][None])

        reward_shaped = np.concatenate(reward_shaped_list).mean(0)
        # plot_3d(env, reward_shaped)
        # reward_shaped /= np.abs(reward_shaped[0])
        plot_value_pred_map(env, reward_shaped, contain_goal_value=True)
        plt.show()
        
        # M = np.concatenate(M_list).mean(0)
        # if rep == "MER":
        #     M = np.log(M)
        # plt.imshow(M)
        # plt.show()


        



    

if __name__ == '__main__':
    app.run(main)

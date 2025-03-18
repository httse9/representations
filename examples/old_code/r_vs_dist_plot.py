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

flags.DEFINE_string('env', 'maxent_empty', 'Environment to run.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')
flags.DEFINE_float('lamb', 1, 'Discount factor to use for SR.')

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

    SR = compute_SR(env, gamma=FLAGS.gamma)
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
        DR = compute_MER(env, lamb=FLAGS.lamb)
    else:
        DR = compute_DR(env, lamb=FLAGS.lamb)

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

    path = 'minigrid_basics/plots/r_vs_dist/'

    ##### 
    # Visualize shaped reward
    #####
    SR_reward = SR_aux_reward(env)
    MER_reward = DR_MER_aux_reward(env, mode='DR')

    directions =np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ])


    def compute_r_diff_vs_dist(env, reward):
        terminal_idx = env.terminal_idx[0]
        goal_pos = env.state_to_pos[terminal_idx]
        dists = []
        diffs = []

        dist_diffs = {}

        for i in range(env.num_states): # enumerate over all states
            pos = env.state_to_pos[i]
            dist = np.abs(np.array(goal_pos) - np.array(pos)).sum()

            n_diff = []
            for dir in directions:
                n_pos = pos + dir   # neighbor position

                try:
                    # print("  ", neighbor)
                    j = env.pos_to_state[n_pos[0] + n_pos[1] * env.width]
                    if j >= 0:
                        n_diff.append(np.abs(reward[i] - reward[j]))
                except:
                    pass

            dists.append(dist)
            diffs.append(np.mean(n_diff))

            try:
                dist_diffs[dist].append(np.mean(n_diff))
            except:
                dist_diffs[dist] = [np.mean(n_diff)]

        dists = []
        diffs = []        
        for k in dist_diffs.keys():
            dists.append(k)
            diffs.append(np.mean(dist_diffs[k]))

        return dists, diffs

    SR_dists, SR_diffs = compute_r_diff_vs_dist(env, SR_reward)
    plt.scatter(SR_dists, SR_diffs, label="SR")

    MER_dists, MER_diffs = compute_r_diff_vs_dist(env, MER_reward)
    plt.scatter(MER_dists, MER_diffs, label="DR")

    plt.xlabel("Distance to Terminal State")
    plt.ylabel("Average Diff in Neighboring Rewards")
    plt.title(env_id)
    plt.legend()
    plt.savefig(join(path, f"{env_id}_SR_vs_DR.png"))

    plt.show()


    for lamb in [1, 2, 3, 5]:
        FLAGS.lamb = lamb
        DR_reward = DR_MER_aux_reward(env, mode='DR')

        dists, diffs = compute_r_diff_vs_dist(env, DR_reward)

        plt.scatter(dists, diffs, label=f"DR_{lamb}")

    plt.title(env_id)
    plt.legend()
    plt.savefig(join(path, f"{env_id}_DR_lamb.png"))
    plt.show()

    for gamma in [0.1, 0.5, 0.9, 0.99]:
        FLAGS.gamma = gamma
        SR_reward = SR_aux_reward(env)

        dists, diffs = compute_r_diff_vs_dist(env, SR_reward)

        plt.scatter(dists, diffs, label=f"SR_{gamma}")

    plt.title(env_id)
    plt.legend()
    plt.savefig(join(path, f"{env_id}_SR_gamma.png"))


    plt.show()

if __name__ == '__main__':
    app.run(main)

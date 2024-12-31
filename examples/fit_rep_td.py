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

# experiment flags
flags.DEFINE_integer('n_episodes', 100, 'Number of episodes to collect data for.')
flags.DEFINE_float('lr', 0.3, 'Learning rate.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def collect_data(env, n_episodes=100):
    data = []  # store transitions to be used for SR/DR/MER fitting

    for e in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.random.choice(env.num_actions)     # uniform random policy
            ns, r, done, d = env.step(a)
            terminated = d['terminated']

            transition = (s['state'], r, ns['state'])
            data.append(transition)

            if terminated:
                r_terminal = env.reward()
                data.append((ns['state'], r_terminal, None))    # terminal state has no next state, so None

            s = ns
        
    return data


def fit_rep_TD(env, data, mode="SR", alpha=0.03):
    """
    Simply run uniform random policy
    """
    if mode == "SR":
        ground_truth_M = compute_SR(env)
    elif mode == "DR":
        ground_truth_M = compute_DR(env)
    elif mode == "MER":
        ground_truth_M = compute_MER(env)
    else:
        raise ValueError(f"{mode} not recognized.")

    n_states = env.num_states
    n_actions = env.num_actions

    if mode == "SR":
        D = np.zeros((n_states, n_states))
    elif mode == "DR":
        D = np.eye(n_states)
    elif mode == "MER":
        D = np.eye(n_states) * n_actions


    mae_list = []       # max absolute error
    mse_list = []       # mean squared error

    for n in range(100):   # repeat
        D_old = D.copy()
        for (s, r, ns) in data:
            
            if ns is None:  # terminated
                if mode == 'SR':
                    D[s, s] = 1
                elif mode == "DR":
                    D[s, s] = np.exp(r)
                elif mode == "MER":
                    D[s, s] = np.exp(r) * n_actions
                continue

            # normal transition
            indicator = np.zeros(n_states)
            indicator[s] = 1

            if mode == "SR":
                D[s] += alpha * (indicator + FLAGS.gamma * D[ns] - D[s])
            elif mode == "DR":
                D[s] += alpha * ((indicator + D[ns]) * np.exp(r) - D[s])
            elif mode == "MER":
                D[s] += alpha * ((indicator + D[ns]) * np.exp(r) * n_actions - D[s])

        mae = np.abs(D - D_old).max()
        mae_list.append(mae)

        mse = ((D - D_old) ** 2).mean()
        mse_list.append(mse)
    
    return ground_truth_M, D, mae_list, mse_list



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
    env = gym.make(env_id, seed=FLAGS.seed)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # learn representation by TD
    data = collect_data(env, FLAGS.n_episodes)

    reps = ["SR", "MER"]
    Ms = []
    maes = []
    mses = []
    for rep in reps:
        _, M, mae, mse = fit_rep_TD(env, data, rep, FLAGS.lr)

        Ms.append(M)
        maes.append(mae)
        mses.append(mse)

    for rep, M, mae, mse in zip(reps, Ms, maes, mses):

        exp_name = [rep, FLAGS.n_episodes, FLAGS.lr, FLAGS.seed]
        exp_name = [str(x) for x in exp_name]
        exp_name = '-'.join(exp_name) + ".pkl"
        path = join("minigrid_basics", "experiments", "fit_rep_td", env.unwrapped.spec.id,)
        os.makedirs(path, exist_ok=True)

        data_dict = dict(
            mae = mae,
            mse = mse,
            M = M
        )

        with open(join(path, exp_name), "wb") as f:
            pickle.dump(data_dict, f)


    

if __name__ == '__main__':
    app.run(main)

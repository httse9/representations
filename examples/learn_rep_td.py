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

# experiment flags
flags.DEFINE_string('representation', 'SR', 'The representation to use for reward shaping.')
flags.DEFINE_float('lr', 0.3, 'Learning rate.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def compute_rep_TD(env, mode="SR", alpha=0.03):
    """
    Simply run uniform random policy
    """
    if mode == "SR":
        ground_truth_M = compute_SR(env)
    elif mode == "DR":
        ground_truth_M = compute_DR(env)
    elif mode == "MER":
        ground_truth_M = compute_MER(env)
        # print(ground_truth_M[78])
        # quit()
    else:
        raise ValueError(f"{mode} not recognized.")

    n_states = env.num_states
    n_actions = env.num_actions

    D = np.zeros((n_states, n_states))
    mae_list = []       # max absolute error
    mse_list = []       # mean squared error

    for e in range(20000):   # enumerate over episodes

        s = env.reset()
        done = False
        while not done:
            a = np.random.choice(n_actions)     # uniform random policy
            ns, r, done, d = env.step(a)
            terminated = d['terminated']

            # if mode == "SR":
            #     alpha = 1 / np.sqrt(e + 1)
            # if mode == "DR":
            #     alpha = 1 / np.sqrt(e + 1) * 3      # decaying learning rate to speed up learning
            # if mode == "MER":
            #     alpha = 1 / np.sqrt(e + 1) * 2


            indicator = np.zeros(n_states)
            indicator[s['state']] = 1

            if mode == "SR":
                D[s['state']] += alpha * (indicator + FLAGS.gamma * D[ns['state']] - D[s['state']])
            elif mode == "DR":
                D[s['state']] += alpha * ((indicator + D[ns['state']]) * np.exp(r) - D[s['state']])
            elif mode == "MER":
                D[s['state']] += alpha * ((indicator + D[ns['state']]) * np.exp(r) * n_actions - D[s['state']])

            if terminated:
                r_terminal = env.reward()
                if mode == 'SR':
                    D[ns['state'], ns['state']] = 1
                    # D[ns['state'], ns['state']] += alpha * (1 - D[ns['state'], ns['state']])
                elif mode == "DR":
                    D[ns['state'], ns['state']] = np.exp(r_terminal)
                    # D[ns['state'], ns['state']] += alpha * (np.exp(r_terminal) - D[ns['state'], ns['state']])
                elif mode == "MER":
                    D[ns['state'], ns['state']] = np.exp(r_terminal) * n_actions
                    # D[ns['state'], ns['state']] += alpha * (np.exp(r_terminal) * n_actions - D[ns['state'], ns['state']])

            s = ns
        
        mae = np.abs(D - ground_truth_M).max()
        mae_list.append(mae)

        mse = ((D - ground_truth_M) ** 2).mean()
        mse_list.append(mse)
    
    return mae_list, mse_list



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

    env_eval = gym.make(env_id, seed=FLAGS.seed)
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # learn representation by TD
    mae, mse = compute_rep_TD(env, mode=FLAGS.representation, alpha=FLAGS.lr)


    exp_name = [FLAGS.representation, FLAGS.lr, FLAGS.seed]
    exp_name = [str(x) for x in exp_name]
    exp_name = '-'.join(exp_name) + ".pkl"
    path = join("minigrid_basics", "experiments", "learn_rep_td", env.unwrapped.spec.id,)
    os.makedirs(path, exist_ok=True)

    data_dict = dict(
        mae = mae,
        mse = mse
    )

    with open(join(path, exp_name), "wb") as f:
        pickle.dump(data_dict, f)


    

if __name__ == '__main__':
    app.run(main)

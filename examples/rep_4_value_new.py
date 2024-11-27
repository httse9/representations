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
from tqdm import tqdm

from minigrid_basics.custom_wrappers import coloring_wrapper
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.envs import maxent_mon_minigrid
from minigrid_basics.examples.rep_utils import *

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'maxent_high_low', 'Environment to run.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')
flags.DEFINE_float('VI_step_size', 0.001, 'step size for value iteration.')

flags.DEFINE_integer('max_iter', 15000, 'Maximum number of iterations.')
flags.DEFINE_integer('log_interval', 100, 'Log interval.')

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')

# experiment flags
flags.DEFINE_boolean('fit_optimal_V', False, 'Run experiment to fit optimal V.')
flags.DEFINE_boolean('learn_V_TD', False, 'Run experiment to use representations to learn V by TD.')

# plot flag
flags.DEFINE_boolean('plot', False, 'When True, plot results of experiments. When False, run the experiment.')
flags.DEFINE_boolean('plot_each', False, '?')

##############################
### Functions for fitting optimal V (START)
##############################

def fit_weight(env, rep, V_target, lr, max_iter=15000, log_interval=100):
    """
    Fix weight for true value function using representation.

    rep: matrix of representations, each row is the representation of one state.
        - Does not contain representation of terminal states.
    V_target: target state values
        - Does not contain value of terminal states.
    lr: learning rate
    """
    # init weights
    weight = np.zeros((rep.shape[1]))

    def compute_mse():
        # return mse between current prediction and V target
        return ((V_target - rep @ weight) ** 2).mean()
    
    mse_list = [compute_mse()]

    pbar = tqdm(total=max_iter)
    for n in range(max_iter):
        weight += lr * rep.T @ (V_target - rep @ weight)

        if (n + 1) % log_interval == 0:
            mse = compute_mse()

            pbar.set_description(f"MSE: {mse}")
            pbar.update(100)
            mse_list.append(mse)

    return weight, mse_list

def fit_weight_lrs(env, rep_name, lrs, max_iter=15000, log_interval=100, verbose=True):
    if rep_name not in ["SR", "DR", "MER"]:
        raise ValueError(f"Representation {rep_name} not recognized.")
    path = join("minigrid_basics", "examples", "rep_plots", "rep_fit", env.unwrapped.spec.id)
    rep_path = join(path, rep_name)
    os.makedirs(rep_path, exist_ok=True)
    
    P = env.transition_probs
    nonterminal_idx = (P.sum(-1).sum(-1) != 0)  #idx of non-terminal states

    # compute optimal V
    V_optimal = value_iteration(env, gamma=FLAGS.gamma)
    V_optimal = V_optimal[nonterminal_idx]

    # compute representation
    rep = get_representation(env, rep_name, gamma=FLAGS.gamma)
    rep = rep[nonterminal_idx]
    if rep_name in ["DR", "MER"]:
        rep = process_DR_or_MER(rep)

    for lr in lrs:
        if verbose:
            print(">> Learning rate:", lr)

        # no need random seed cause no stochasticity
        w_fit, mse = fit_weight(env, rep, V_optimal, lr, max_iter=max_iter, log_interval=log_interval)

        data_dict = dict(
            w_fit = w_fit,
            mse = mse
        )

        with open(join(rep_path, f"{lr}.pkl"), "wb") as f:
            pickle.dump(data_dict, f)
    
##############################
### Functions for fitting optimal V (END)
##############################

# main
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
    env = gym.make(env_id)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    ##############################
    ### Experiment to use reprsentations
    # to fit optimal value function
    ##############################
    if FLAGS.fit_optimal_V:
        if FLAGS.plot:
            pass

        lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        fit_weight_lrs(env, "SR", lrs)



if __name__ == '__main__':
    app.run(main)

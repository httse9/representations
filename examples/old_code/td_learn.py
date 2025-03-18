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

from minigrid_basics.examples.td_learner import TD_Learner
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.envs import maxent_mon_minigrid

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'maxent_high_low', 'Environment to run.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')

flags.DEFINE_boolean('plot', False, 'When True, plot results of experiments. When False, run the experiment.')


def construct_value_pred_map(env, value_prediction, contain_goal_value=False):
    """
    Take the vector of predicted values, and visualize in the environment map.

    Params:
    value_prediction: vector of predicted state values, shape (S)
    contain_goal_value: whether the input value_prediction contains the value \ 
        prediction of goal values. If False, 
    """
    state_num = 0
    value_pred_map = np.zeros(env.reward_grid.shape) - float('-inf')

    for i in range(env.height):
        for j in range(env.width):
            if env.reward_grid.T[i, j] == 1:
                # skip wall
                continue

            if not contain_goal_value:
                if env.raw_grid.T[i, j] == 'g':
                    value_pred_map[i, j] = env.reward_grid.T[i, j]
                    continue


            value_pred_map[i, j] = value_prediction[state_num]
            state_num += 1

    return value_pred_map

def plot_value_pred_map(env, value_prediction, contain_goal_value=False, v_range=(None, None)):
    map = construct_value_pred_map(env, value_prediction, contain_goal_value=contain_goal_value)

    vmin, vmax = v_range
    plt.imshow(map, vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])

    # matplotlib.rcParams.update({'font.size': 5})
    for (j, i), label in np.ndenumerate(map):
        if not np.isinf(label):
            plt.text(i, j, np.round(label, 1), ha='center', va='center', color='white', \
                     fontsize= 'xx-small')

def plot_td_learn_data(env, ):
    path = join("minigrid_basics", "examples", "rep_plots", "td_learn", env.unwrapped.spec.id,)

    # reps = ["MER", "MER_tabular", "DR_tabular", "DR", "SR", "SR_tabular"]
    reps = ["SR", "SR_tabular", "DR", "DR_tabular", "MER", "MER_tabular"]
    for i, rep in enumerate(reps):
        rep_path = join(path, rep)

        fnames = os.listdir(rep_path)
        fnames = [d for d in fnames if "pkl" in d]    # filter non data files

        lrs = [d.split(".pkl")[0] for d in fnames]       # strings

        all_return_list = []

        for j, (lr, fname) in enumerate(zip(lrs, fnames)):
            with open(join(rep_path, fname), "rb") as f:
                data_dict = pickle.load(f)
            
            mse_list = []
            return_list = []        # contains returns for all seeds [[ret seed 0], [ret seed 1], ...]
            for data in data_dict['data']:
                mse_list.append(data['mse'])
                return_list.append(data['return'])

            all_return_list.append(return_list)

            plt.subplot(1, 2, 1)
            plot_mean_and_conf_interval(data['n_iter'], mse_list, label=lr)
            plt.subplot(1, 2, 2)
            plot_mean_and_conf_interval(data['n_iter'], return_list, label=lr)
        plt.legend()
        plt.tight_layout()
        plt.show()

    #     n_evals = len(all_return_list[0][0])
    #     mean_returns = [np.mean(ret, 0)[-n_evals // 10: ].mean() for ret in all_return_list]
    #     print(rep, mean_returns)

    #     best_lr_idx = np.argmax(mean_returns)

    #     return_list = all_return_list[best_lr_idx]
    #     plot_mean_and_conf_interval(data['n_iter'], return_list, label=rep)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

def plot_td_learn_best_lr(env):
    path = join("minigrid_basics", "examples", "rep_plots", "td_learn", env.unwrapped.spec.id,)

    reps = ["SR", "SR_tabular", "DR", "DR_tabular", "MER", "MER_tabular"]


    if FLAGS.env == "maxent_maze":
        best_lr = [0.01, 0.3, 200000000, 1, 3000000, 1]


    for rep, lr in zip(reps, best_lr):
        rep_path = join(path, rep)
        with open(join(rep_path, f"{lr}.pkl"), "rb") as f:
            data_dict = pickle.load(f)

        return_list = []
        for data in data_dict['data']:
            return_list.append(data['return'])

        plot_mean_and_conf_interval(data['n_iter'], return_list, label=rep)
    
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, 20000])
    plt.show()

            
def plot_mean_and_conf_interval(x, y, label):

    """
    x: x axis
    y: data, each row contains one trial
    """
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    n_trials = np.array(y).shape[0]

    y_interval = 1.96 / np.sqrt(n_trials) * y_std

    plt.plot(x, y_mean, label=label)
    plt.fill_between(x, y_mean - y_interval, y_mean + y_interval, alpha=0.2)

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

    env_eval = gym.make(env_id)
    env_eval =  maxent_mdp_wrapper.MDPWrapper(env_eval)

    if FLAGS.plot:
        plot_td_learn_data(env)
        plot_td_learn_best_lr(env)
        quit()

    td_learner = TD_Learner(env, env_eval, 'DR', tabular=False, v_lr=3_000_000)

    print(td_learner.F[-2])

    for seed in range(5):
        data = td_learner.learn(200000, log_interval=10000, seed=seed) #(200000)

    # V_SR = td_learner.compute_optimal_V_SR()
    # plt.subplot(1, 3, 1)
    # plot_value_pred_map(env, V_SR, True)
    # # plt.show()

    # V_DR = td_learner.compute_optimal_V_DR()
    # plt.subplot(1, 3, 2)
    # plot_value_pred_map(env, V_DR, True)
    # plt.show()

    # V_MER = td_learner.compute_optimal_V_MER()
    # plt.subplot(1, 3, 3)
    # plot_value_pred_map(env, V_MER, True)
    # plt.show()

    # V = td_learner.compute_optimal_V()
    # plot_value_pred_map(env, V, True)
    # plt.show()

    # plt.plot(data['n_iter'], data['return'])
    # plt.show()

    # plt.plot(data['n_iter'], data['mse'])
    # plt.show()

    # TODO:
    # 1. Test SR, DR, MER
    #   - SR tabular (done, lr 0.1 or 1)
    #   - SR feat (done, lr 0.01, need smaller lr)
    #   - DR tabular (done, lr 1)
    #   - DR feat (need lr of 100_000_000 to make it work)
    #       - Problem: magnitude of entries differ by too much, too small, require extremely large lr to work.
    #       - Can converge pretty well with lr 100_000_000
    #   - DR large lambda 
    #       - Larger lambda works, but policy is gonna be more random
    #   - DR log feat
    #       - may or may not work, but not mathematically sound.
    #   - MER tabular (done, lr 1)
    #   - MER feat (need lr of 3_000_000.)
    #   - MER log feat
    # 3. Why is expected return of policy not equal to optimal value of start state?


    """
    high_low
    SR:
    DR: works with 100_000_000 (but unstable, maybe smaller?)
    MER: works with 3_000_000

    reward_road
    SR:
    DR: 1e23, 1e24 (sometimes work, very very unstable)
    MER:

    New trick:
    - adaptive learning rate (works well for maze 2,)
        - works extremely well for reward road. 
        - needs to make sure that numerical stability term for lr is smaller than max change.

    Maybe use something like adam?
    """

    # SR, DR works with large lr (0.1).
    # DR needs much larger number of steps, but might need even larger lr

if __name__ == '__main__':
    app.run(main)

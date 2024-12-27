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
flags.DEFINE_string('representation', 'baseline', 'The representation to use for reward shaping.')
flags.DEFINE_float('lr', 0.3, 'Learning rate for Q-Learning.')
# flags.DEFINE_float('r_orig_weight', 1.0, 'Learning rate for Q-Learning.')
flags.DEFINE_float('r_shaped_weight', 0.5, 'Learning rate for Q-Learning.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('i_eigen', 0, 'Which eigenvector to use. 0: top eigenvector')


def q_learning(env, env_eval, reward_aux, max_iter=10000, alpha=0.3, log_interval=1000, \
            r_shaped_weight=0.5):
    
    assert 0 <= r_shaped_weight <= 1
    Q = np.zeros((env.num_states, env.num_actions))

    timesteps = []
    ret_evals = []
    s = env.reset()
    for n in range(max_iter):

        if np.random.rand() < 0.05:
            a = np.random.choice(env.num_actions)
        else:
            a = np.argmax(Q[s['state']])

        ns, r, done, d = env.step(a)
        terminated = d['terminated']

        # shaped reward
        r = r * (1 - r_shaped_weight) + reward_aux[s['state']] * r_shaped_weight

        Q[s['state'], a] += alpha * (r + FLAGS.gamma * (1 - int(terminated)) * Q[ns['state']].max()  - Q[s['state'], a])

        if done:
            s = env.reset()
        else:
            s = ns

        if (n + 1) % log_interval == 0:
            pi_new = (Q == Q.max(1, keepdims=True)).astype(float)
            pi_new /= pi_new.sum(1, keepdims=True)  
            ret_eval = eval_policy(env_eval, pi_new, gamma=1, n_episodes=1)    # env and policy both no stochasticity
            
            timesteps.append(n + 1)
            ret_evals.append(ret_eval)

            # Q_max = Q.max(1)
            # plot_value_pred_map(env, Q_max, contain_goal_value=True)
            # plt.show()

    return Q, timesteps, ret_evals

def eval_policy(env, pi, gamma=0.99, n_episodes=10):
    """
    Evaluate return of policy pi
    """
    return_list = []
    for n in range(n_episodes):

        s = env.reset()

        done = False
        discount = 1
        episode_return = 0

        while not done:
            a = np.random.choice(env.num_actions, p=pi[s['state']])

            s, r, done, _ = env.step(a)
            # print(s)

            episode_return += discount * r
            discount *= gamma

        return_list.append(episode_return)

    return np.mean(return_list)

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

def load_setting(path, rep, i_eigen, r_shaped_weight, lr, n_seeds=20):

    
    datas = []
    for seed in range(1, n_seeds + 1):

        # construct file name
        exp_name = [rep, i_eigen, r_shaped_weight, lr, seed]
        exp_name = [str(x) for x in exp_name]
        exp_name = '-'.join(exp_name) 

        with open(join(path, exp_name + ".pkl"), "rb") as f:
            data = pickle.load(f)

        datas.append(data['perf'])

    # plot_mean_and_conf_interval(data['t'], datas, label=exp_name)
    return data['t'], datas, exp_name


def plot(env_id):


    path = join("minigrid_basics", "experiments", "reward_shaping", env_id)


    # best for baseline
    x, y, name = load_setting(path, "baseline", 0, 0.0, 0.3)
    plot_mean_and_conf_interval(x, y, label=name, color="black", alpha=1)


    r_shaped_weights = [0.25, 0.5, 0.75, 1.0]
    def construct_alphas(settings):
        n = len(settings)
        alphas = np.linspace(0.5, 1, n + 1)[1:]
        return alphas
    
    def construct_colors(settings, rgb=0):
        n = len(settings)
        c_values = np.linspace(0.1, 1, n)
        
        colors = np.zeros((n, 3))
        colors[:, rgb] = c_values
        return colors


    

    for rep, best_lr, color in zip(["SR", "MER"], [1.0, 1.0], ['red', 'blue']):
        for i_eigen in [0]:
            for r_shaped_weight, alpha in zip(r_shaped_weights, construct_alphas(r_shaped_weights)):

                # for lr in [0.1, 0.3, 1.0]:
                    # x, y, name = load_setting(path, rep, i_eigen, r_shaped_weight, lr)
                    # plot_mean_and_conf_interval(x, y, label=name)

                x, y, name = load_setting(path, rep, i_eigen, r_shaped_weight, best_lr)
                plot_mean_and_conf_interval(x, y, label=name, color=color, alpha=alpha)

    plt.legend()
    plt.ylim([-300, None])
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Undiscounted Return")
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

    if FLAGS.plot:
        plot(env_id)
        quit()

    ##############################
    ### Make env
    ##############################
    # do not treat goal as absorbing
    env = gym.make(env_id, seed=FLAGS.seed)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    env_eval = gym.make(env_id, seed=FLAGS.seed)
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)
    

    # SR_e0 = SR_aux_reward(env)
    # DR_e0 = MER_aux_reward(env)

    ### Plot shaped reward of the DR and the SR
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # SR_value_map = construct_value_pred_map(env, SR_e0, contain_goal_value=True).T
    # SR_value_map[np.isinf(SR_value_map)] = np.median(SR_value_map[~np.isinf(SR_value_map)])

    # get rid of walls
    # SR_value_map = SR_value_map[1:-1, 1:-1]

    # SR_value_map = np.rot90(SR_value_map, k=2).T

    # x, y = SR_value_map.shape
    # x, y = np.meshgrid(range(x), range(y))
    # ax.plot_surface(x, y, SR_value_map, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.title.set_text("SR")


    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # DR_value_map = construct_value_pred_map(env, DR_e0, contain_goal_value=True).T
    # DR_value_map[np.isinf(DR_value_map)] = np.median(DR_value_map[~np.isinf(DR_value_map)])
    # DR_value_map = DR_value_map[1:-1, 1:-1]
    # DR_value_map = np.rot90(DR_value_map, k=2).T
    # ax.plot_surface(x, y, DR_value_map, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.title.set_text("MER")
    # plt.show()

    # quit()

    # plt.subplot(1, 2, 1)
    # plot_value_pred_map(env, SR_e0, contain_goal_value=True)
    # plt.subplot(1, 2, 2)
    # plot_value_pred_map(env, DR_e0, contain_goal_value=True)
    # plt.show()
    # quit()

    
    if FLAGS.representation == 'SR':
        reward_shaped = SR_aux_reward(env, i=FLAGS.i_eigen)
    elif FLAGS.representation == "DR":
        reward_shaped = DR_MER_aux_reward(env, i=FLAGS.i_eigen, mode="DR")
    elif FLAGS.representation == 'MER':
        reward_shaped = DR_MER_aux_reward(env, i=FLAGS.i_eigen, mode="MER")
    elif FLAGS.representation == 'baseline':
        # no reward shaping
        reward_shaped = np.zeros((env.num_states))
        FLAGS.r_shaped_weight = 0.

    # set seed
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    Q, t, performance = q_learning(env, env_eval, reward_shaped, max_iter=20000, log_interval=50, \
            alpha=FLAGS.lr, r_shaped_weight=FLAGS.r_shaped_weight)
    
    # plt.plot(t, performance)
    # plt.show()

    exp_name = [FLAGS.representation, FLAGS.i_eigen, FLAGS.r_shaped_weight, FLAGS.lr, FLAGS.seed]
    exp_name = [str(x) for x in exp_name]
    exp_name = '-'.join(exp_name) + ".pkl"
    path = join("minigrid_basics", "experiments", "reward_shaping", env.unwrapped.spec.id,)
    os.makedirs(path, exist_ok=True)

    data_dict = dict(
        t=t,
        perf=performance,
        Q=Q
    )

    with open(join(path, exp_name), "wb") as f:
        pickle.dump(data_dict, f)


    

if __name__ == '__main__':
    app.run(main)

# # treat goal as absorbing state
# env_goal_absorbing = gym.make(env_id)
# env_goal_absorbing =  maxent_mdp_wrapper.MDPWrapper(env_goal_absorbing, goal_absorbing=True)

# ###  SR absorbing
# SR = compute_SR(env_goal_absorbing)
# if not np.allclose(SR, SR.T):
#     SR = (SR + SR.T) / 2

# lamb, e = np.linalg.eig(SR)
# idx = lamb.argsort()
# e = e.T[idx[::-1]]  
# e0 = np.real(e[1])      # second largest eigenvector for absorbing

# e0 = - np.abs(e0[terminal_idx] - e0)      # shaped reward
# e0 /= np.abs(e0).max()  # normalize

# plt.subplot(1, 2, 2)
# plot_value_pred_map(env, e0, contain_goal_value=True)
# plt.show()


# # handle negative entries in DR #1
# # replace with smallest absolute value of DR
# # there exists some black holes (reject, replaced value is too small)
# min_abs_v = np.abs(e0).min()
# e0[e0 < 0] = min_abs_v

# # handle negative entries in DR #2   (reject, removes too much information, changes all values)
# # shift whole matrix value upwards
# min_abs_v = np.abs(e0).min()
# if (e0 < 0).any():
#     e0 -= e0.min()
#     e0 += min_abs_v

# # handle #3 (reject, values very different from neighbors)
# # simply flip negative entries
# e0[e0 < 0] *= -1

# # handle #4 simple interpolate
# for i in range(len(e0)):
#     if e0[i] < 0:
#         values = []
#         # get prev value
#         try:
#             values.append(e0[i-1])
#         except:
#             pass
#         # get next value
#         try:
#             values.append(e0[i + 1])
#         except:
#             pass
#         e0[i] = np.mean(values)

# # handle #5 careful interpolate
# # average over values of neighboring grids
# directions =np.array([
#     [1, 0],
#     [0, 1],
#     [-1, 0],
#     [0, -1]
# ])
# for i in range(len(e0)):
#     if e0[i] < 0:
#         pos = np.array(env.state_to_pos[i])
#         neighbor_values = []
#         for d in directions:
#             try:
#                 neighbor = pos + d
#                 # print("  ", neighbor)
#                 j = env.pos_to_state[neighbor[0] + neighbor[1] * env.width]
#                 if j >= 0:
#                     neighbor_values.append(e0[j])
#             except:
#                 pass

#         e0[i] = np.mean(neighbor_values)

# e0 = np.log(e0)     # take log
# e0 = - np.abs(e0[terminal_idx] - e0)      # shaped reward
# e0 /= np.abs(e0).max()  # normalize

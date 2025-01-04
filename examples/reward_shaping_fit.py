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

### experiment flags
flags.DEFINE_string('representation', 'SR', 'The representation to use for reward shaping.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

# learning representation related
# flags.DEFINE_float('rep_learn_lr', 0.1, 'Learning rate for learning representations using TD.')
flags.DEFINE_integer('n_episodes', 100, 'Number of episodes to use for learning representations.')

# reward shaping related
flags.DEFINE_integer('i_eigen', 0, 'Which eigenvector to use. 0: top eigenvector')
flags.DEFINE_float('r_shaped_weight', 0.5, 'Learning rate for Q-Learning.')
flags.DEFINE_float('lr', 0.1, 'Learning rate for Q-Learning.')

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
    n_states = env.num_states
    n_actions = env.num_actions

    D = np.zeros((n_states, n_states))

    for _ in range(300):   # repeat

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
    
    return D

def q_learning(env, env_eval, reward_aux, max_iter=10000, alpha=0.3, log_interval=1000, \
            r_shaped_weight=0.5):
    
    assert 0 <= r_shaped_weight <= 1
    Q = np.zeros((env.num_states, env.num_actions))

    R = env.rewards
    R_orig_max = np.abs(R).max()

    timesteps = []
    ret_evals = []
    s = env.reset()
    for n in range(max_iter):

        if np.random.rand() < 0.05:     # epsilon greedy
            a = np.random.choice(env.num_actions)
        else:
            a = np.argmax(Q[s['state']])

        ns, r, done, d = env.step(a)
        terminated = d['terminated']

        # shaped reward
        # convex combination of normalized orig and shaped rewards
        r = r / R_orig_max * (1 - r_shaped_weight) + reward_aux[s['state']] * r_shaped_weight

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

    data = collect_data(env, FLAGS.n_episodes)
    SR = fit_rep_TD(env, data, mode="SR", alpha=0.3)
    
    if not np.allclose(SR, SR.T):   # handle assymetry, avoid imaginary numbers
        SR = (SR + SR.T) / 2

    # eigendecomposition
    lamb, e = np.linalg.eig(SR)
    idx = lamb.argsort()
    e = e.T[idx[::-1]]
    e0 = np.real(e[i])  # largest eigenvector

    e0 = - np.abs(e0[terminal_idx] - e0)      # shaped reward
    e0 /= np.abs(e0).max()  # normalize

    return e0, SR

def DR_MER_aux_reward(env, i=0):
    # terminal_idx = np.where(~env.nonterminal_idx)[0][0]
    terminal_idx = env.terminal_idx[0]
    data = collect_data(env, FLAGS.n_episodes)

    DR = fit_rep_TD(env, data, FLAGS.representation, alpha=0.3)

    if not np.allclose(DR, DR.T): # handle asymmetry
        DR = (DR + DR.T) / 2

    lamb, e = np.linalg.eig(DR)
    idx = lamb.argsort()
    e = e.T[idx[::-1]]
    e0 = np.real(e[i])      # get i-th eigenvector

    if (e0 < 0).astype(int).sum() > (e0 > 0).astype(int).sum():
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

    for _ in range(env.num_states):  # repeatedly try to interpolate value from neighbors
        for i in range(len(e0_copy)):   # enumerate over all states
            if e0[i] <= 0:      # if value is invalid, try to replace by mean of neighbors

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
                if len(neighbor_values) > 0:    # successful
                    e0_copy[i] = np.mean(neighbor_values)
                    e0[i] = 1
    e0 = e0_copy

    e0 = - np.abs(e0[terminal_idx] - e0)      # shaped reward

    if np.abs(e0).max() > 0:
        e0 /= np.abs(e0).max()  # normalize

    return e0, DR


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
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    env_eval = gym.make(env_id, seed=FLAGS.seed)
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)

    # set seed
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    
    if FLAGS.representation == 'SR':
        reward_shaped, M = SR_aux_reward(env, i=FLAGS.i_eigen)
    elif FLAGS.representation == "DR":
        reward_shaped, M = DR_MER_aux_reward(env, i=FLAGS.i_eigen)
    elif FLAGS.representation == 'MER':
        reward_shaped, M = DR_MER_aux_reward(env, i=FLAGS.i_eigen)
    else:
        raise ValueError()
    
    if FLAGS.r_shaped_weight == 0: # equivalent to no reward shaping..
        quit()

    Q, t, performance = q_learning(env, env_eval, reward_shaped, max_iter=50000, log_interval=10, \
            alpha=FLAGS.lr, r_shaped_weight=FLAGS.r_shaped_weight)
    
    
    exp_name = [FLAGS.representation, FLAGS.n_episodes, FLAGS.i_eigen, FLAGS.r_shaped_weight, FLAGS.lr, FLAGS.seed]
    exp_name = [str(x) for x in exp_name]
    exp_name = '-'.join(exp_name) + ".pkl"
    path = join("minigrid_basics", "experiments", "reward_shaping_fit", env.unwrapped.spec.id,)
    os.makedirs(path, exist_ok=True)

    data_dict = dict(
        t=t,
        perf=performance,
        Q=Q,
        M=M,
        reward_shaped = reward_shaped
    )

    with open(join(path, exp_name), "wb") as f:
        pickle.dump(data_dict, f)



if __name__ == '__main__':
    app.run(main)

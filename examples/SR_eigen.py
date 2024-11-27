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
import scipy

from minigrid_basics.custom_wrappers import coloring_wrapper
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.envs import maxent_mon_minigrid


"""
Compare how we handle terminal state in the successor representation.
Two possibilities:
1. In transition prob matrix P, all 0 entries for terminal state
2. In P, terminal state transition to itself with prob 1
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'maxent_fourrooms', 'Environment to run.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')


def compute_SR(pi, P, gamma):
    """
    Compute the successor representation (SR).
    pi: policy with shape (S, A)  
    P: transition probability matrix with shape (S, A, S)
    gamma: discount factor, scalar
    """
    # compute P_pi, prob of state transition under pi
    P_pi = (P * pi[..., None]).sum(1)

    # compute SR
    n_states = P_pi.shape[0]
    SR = np.linalg.inv(np.eye(n_states) - gamma * P_pi)
    return SR

def eigen(M):
    """
    Take eigendecomposition of input matrix M
    Sort eigenvalues and eigenvectors (descending)
    Return eigenvalues (vector), eigenvectors (rows of matrix)
    """
    e, v = np.linalg.eig(M)
    idx = e.argsort()

    e = e[idx[::-1]]
    v = v.T[idx[::-1]]

    # return np.real(e), np.real(v)
    return np.imag(e), np.imag(v)
    # return e, v

def is_symmetric(M):
    """
    Return whether M is symmetric
    """
    return np.allclose(M, M.T)

def construct_value_map(env, value_prediction, contain_goal_value=False):
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


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    gin.parse_config_files_and_bindings(
        [os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, '{}.gin'.format(FLAGS.env))],
        bindings=FLAGS.gin_bindings,
        skip_unknown=False)
    env_id = maxent_mon_minigrid.register_environment()

    # get transition prob if goal not absorbing
    env = gym.make(env_id)
    env = maxent_mdp_wrapper.MDPWrapper(env)
    P_non_absorbing = env.transition_probs
    R_non_absorbing = env.rewards

    # get transition prob if goal is absorbing
    env = gym.make(env_id)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)
    P_absorbing = env.transition_probs
    R_absorbing = env.rewards

    # print(P_absorbing.sum(1).sum(1))
    # print(P_non_absorbing.sum(1).sum(1))

    n_states = env.num_states
    n_actions = env.num_actions

    # reward should be the same
    assert np.allclose(R_non_absorbing, R_absorbing)

    # look at rows corresponding to terminal states
    # print(P_non_absorbing[-1])
    # print(P_non_absorbing[-11])

    # print(P_absorbing[-1])
    # print(P_absorbing[-11])

    # look at SR for the two cases
    pi_uniform = np.ones((n_states, n_actions)) / n_actions
    SR_non_absorbing = compute_SR(pi_uniform, P_non_absorbing, FLAGS.gamma)
    SR_absorbing = compute_SR(pi_uniform, P_absorbing, FLAGS.gamma)

    # print(SR_non_absorbing.sum(1))
    # print(SR_absorbing.sum(1))

    ########    (Can Ignore)
    #  SR absorbing should have row same equal for all rows
    #  but numerical issues may cause very small differences
    #  we fix the numerical issue here 
    # Actually, does not affect things much...
    # row_sum_mean = SR_absorbing.sum(1).mean()
    # SR_absorbing[:, -1] -= SR_absorbing.sum(1) - row_sum_mean 
    # print(SR_absorbing.sum(1)[0])
    # print(SR_absorbing.sum(1)[1])
    # print("Row sum All same:", (SR_absorbing.sum(1) == row_sum_mean).all())
    ########


    # plt.subplot(1, 2, 1)
    # plt.imshow(SR_non_absorbing)
    # plt.title("Non-absorbing")
    # plt.subplot(1, 2, 2)
    # plt.imshow(SR_absorbing)
    # plt.title("Absorbing")
    # plt.tight_layout()
    # plt.show()
    # plt.clf()

    print("SR_non_absorbing symmetric:", is_symmetric(SR_non_absorbing))
    print("SR_absorbing symmetric:", is_symmetric(SR_absorbing))

    # if not is_symmetric(SR_non_absorbing):
    SR_non_absorbing_sym = (SR_non_absorbing + SR_non_absorbing.T) / 2
    # if not is_symmetric(SR_absorbing):
    SR_absorbing_sym = (SR_absorbing + SR_absorbing.T) / 2


    # look at eigenvectors
    e_non, v_non = eigen(SR_non_absorbing)
    e_non_sym, v_non_sym = eigen(SR_non_absorbing_sym)
    e_abs, v_abs = eigen(SR_absorbing)
    e_abs_sym, v_abs_sym = eigen(SR_absorbing_sym)
    # print(e_non)
    # print(e_abs)

    plt.plot(e_non, label="Non-absorbing")
    plt.plot(e_non_sym, label="Non-absorbing-sym")
    plt.plot(e_abs, label="Absorbing")
    plt.plot(e_abs_sym, label="Absorbing-sym")
    plt.legend()
    plt.title("Eigenvalues of SR Non-absorbing vs Absorbing")
    # plt.show()
    plt.clf()

    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'figure.figsize': (12, 10)})
    n_eigenvector_plot = 6
    label= ["NonAbsorbing", "NonAbsorbingSym", "Absorbing", "AbsorbingSym"]
    for i, v in enumerate([v_non, v_non_sym, v_abs, v_abs_sym]):
        for j in range(n_eigenvector_plot):

            plt.subplot(len(label), n_eigenvector_plot, i * n_eigenvector_plot + j + 1)
            # plt.axis('off')
            plt.xticks([]),plt.yticks([])

            if i == 0:
                plt.title(f"e{j + 1}")
            if j == 0:
                plt.ylabel(label[i])

            # eigenvector = v[j].reshape(11, 11)
            # eigenvector = np.round(eigenvector, 12)
            # # if eigenvector[0, 0] < 0: # np.abs(eigenvector).sum() < 0: # (eigenvector < 0).all():
            #     # avoid bug of plt when plotting all negative entries
            # eigenvector *= -1
            # if (eigenvector < 0).all():
            #     eigenvector *= -1

            eigenvector = v[j]
            eigenvector = np.round(eigenvector, 12)
            eigenvector = construct_value_map(env, eigenvector, True)

            plt.imshow(eigenvector)

    plt.suptitle("Three Goals")
    plt.tight_layout()
    plt.show()


    


    ###### It is possible to find linear combination of top eigenvectors to make
    # the constant vector
    # v_largest = v_abs[:2].T
    # w = np.linalg.inv(v_largest.T @ v_largest) @ v_largest.T @ np.ones((n_states))
    # print(w)
    # print(w[0] * v_abs[0] + w[1] * v_abs[1])
    ######


    # plt.imshow(v_abs[0].reshape(11, 11))
    # plt.show()

    # print(v_abs[0])
    




if __name__ == '__main__':
    app.run(main)
    # app.run(plot)
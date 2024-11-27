import os

from absl import app
from absl import flags
import gin
import gym
import matplotlib
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


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'maxent_high_low', 'Environment to run.')

flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')

flags.DEFINE_float('VI_step_size', 0.001, 'step size for value iteration.')

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')

# experiment flags
flags.DEFINE_boolean('fit_optimal_V', False, 'Run experiment to fit optimal V.')
flags.DEFINE_boolean('learn_V_TD', False, 'Run experiment to use representations to learn V by TD.')
flags.DEFINE_boolean('eigen_approx_V', False, 'Use eigenvectors of representations as basis fucntions to approximate V optimal.')



# plot flag
flags.DEFINE_boolean('plot', False, 'When True, plot results of experiments. When False, run the experiment.')
flags.DEFINE_boolean('plot_each', False, '?')



"""
TODO:
1. value iteration to compute true value. (done)
2. Simple true value fitting using SR DR MER. Vary different learning rate. Pick best result for each rep. (done)
3. (TD for policy evaluation) + (policy improvement with model) using SR DR MER.
"""


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

def compute_DR(pi, P, R):
    """
    Compute the default representation (DR).
    pi: default policy with shape (S, A)
    P: transition probability matrix with shape (S, A, S)
    R: reward vector with shape (S)
    """
    P_pi = (P * pi[..., None]).sum(1)
    DR = np.linalg.inv(np.diag(np.exp(-R)) - P_pi)
    return DR

def compute_MER(P, R, n_states, n_actions):
    """
    Compute the default representation (DR).
    P: transition probability matrix with shape (S, A, S)
    R: reward vector with shape (S)
    n_states: number of states, scalar
    n_actions: number of actions, scalar
    """
    pi_uniform = np.ones((n_states, n_actions)) / n_actions
    P_pi = (P * pi_uniform[..., None]).sum(1)
    MER = np.linalg.inv(np.diag(np.exp(-R)) / n_actions - P_pi)
    return MER

def process_DR_or_MER(M):
    """
    Since DR and MER values are too small and close, 
    process them before use.
    * Assumes that only contain rows and columns of non-temrinal states.
    After processing, entries dist is roughly normal.

    M: matrix to be processed
    """
    M_processed = M.copy()
    # apply log
    M_processed = np.log(M_processed)
    # normalize
    M_processed /= np.abs(M_processed).max()

    return M_processed

def value_iteration(P, R):
    """
    VI to get ground truth value function
    P: transition prob matrix (S, A, S)
    R: reward vector, (S)
    """
    n_states = R.size
    value = np.zeros((n_states))

    while True:
        value_old = value.copy()

        value = R + FLAGS.gamma * (P @ value).max(1)

        if ((value_old - value) ** 2).sum() < 1e-10:
            break

    return value

def eval_policy(env, pi, n_episodes=10):
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
            discount *= FLAGS.gamma

        return_list.append(episode_return)

    return np.mean(return_list)

def plot_learn_V_TD_result(env):

    path = join("minigrid_basics", "examples", "rep_plots", "rep_learn_td", env.unwrapped.spec.id)
    x = None

    reps = ["SR", "DR", "MER"]
    colors = ["red", "green", "blue", "magenta", "cyan", "black", "brown", "orange", "yellow"]
    markers = ['o', 's', '^', 'h']

    for i, rep in enumerate(reps):
    
        rep_path = join(path, rep)

        fnames = os.listdir(rep_path)
        fnames = [d for d in fnames if "pkl" in d]    # filter non data files

        lrs = [d.split(".pkl")[0] for d in fnames]       # strings
        
        mse_list = []
        return_list = []
        w_list = []

        for j, (lr, fname) in enumerate(zip(lrs, fnames)):
            with open(join(rep_path, fname), "rb") as f:
                data = pickle.load(f)
                mse = data['mse']
                mse_list.append(mse)

                w_list.append(data['w_fit'])
                return_list.append(data['ret'])

            if x is None:
                x = np.array(range(len(mse[0]))) * 100

            if FLAGS.plot_each:
                plt.subplot(1, 2, 1)
                plt.plot(x, mse.mean(0), label=lr, color=colors[j])

                plt.subplot(1, 2, 2)
                plt.plot(x, return_list[-1].mean(0), label=lr, color=colors[j])

        if FLAGS.plot_each:
            for i in range(1, 3):
                plt.subplot(1, 2, i)
                plt.title(rep)
                plt.legend()
            plt.show()
            continue

        mse_list = np.array(mse_list)       # (# lr, # seeds, # logs)
        return_list = np.array(return_list)

        mean_mse_list = mse_list.mean(1)
        best_idx = np.nanargmin(mean_mse_list[:, -1])    # ignore nans

        plt.subplot(1, 2, 1)
        plt.plot(x, mean_mse_list[best_idx], label=rep, color=colors[i])   
        # plt.

        mean_return_list = return_list.mean(1)
        best_idx = np.nanargmax(mean_return_list[:, -1])

        plt.subplot(1, 2, 2)
        plt.plot(x, return_list[best_idx].mean(0), label=rep, color=colors[i])
        # plt.show()

        # best_w_for_reps.append(w_list[best_idx])     

    if not FLAGS.plot_each:
        # plt.ylim([0, 100])
        plt.subplot(1, 2, 1)
        plt.xlabel("Number of iterations")
        plt.ylabel("MSE between Predicted and True V")

        plt.subplot(1, 2, 2)
        plt.xlabel("Number of iterations")
        plt.ylabel("Mean Return")
        plt.legend()

        plt.suptitle(env.unwrapped.spec.id)

        
        plt.show()

def learn_V_TD_per_lr(env, env_eval, rep, rep_name, P, R, V_optimal, lr, n_seeds=5):
    path = join("minigrid_basics", "examples", "rep_plots", "rep_learn_td", env.unwrapped.spec.id)
    os.makedirs(path, exist_ok=True)

    ws = []
    mses = []
    rets = []

    print(">> Learning rate:", lr)
    for i in range(n_seeds):
        np.random.seed(i)
        print(f"  Seed {i + 1}")

        w, mse, ret = TD_eval_model_improv(env, env_eval, rep, P, R, V_optimal, lr)

        ws.append(w)
        mses.append(mse)
        rets.append(ret)

    rep_path = join(path, rep_name)
    os.makedirs(rep_path, exist_ok=True)

    data_dict = dict(
        mse=np.array(mses),
        ret=np.array(rets),
        w_fit = np.array(ws)
    )
    # print(data_dict)
    with open(join(rep_path, f"{lr}.pkl"), "wb") as f:
        pickle.dump(data_dict, f)

def TD_eval_model_improv(env, env_eval, rep_input, P, R, V_optimal, lr, eps=0.01 ,max_iter=5000, n_eval_episodes=1, log_interval=100):
    """
    Use the given representation to learn the optimal value function.
    For policy evaluation, interacts with the env and uses TD.
    For policy improvement, uses the model directly.
    Purpose is to just focus solely on using the given representations
        to learn value functions (V instead of Q).

    Params:
    env: environment
    rep: given representations
    P: transition probability matrix (S, A, S)
    R: reward vector (S)
    V_optimal: optimal value function for computing loss
    lr: learning rate
    eps: for epsilon greedy

    Returns learned value function and policy.
    """
    # init weight
    n_states, n_actions, _ = P.shape
    w = np.random.normal(size=(n_states)) #np.zeros((n_states))

    nonterminal_idx = (P.sum(-1).sum(-1) != 0)
    if rep_input.shape[0] !=  n_states:
        rep = np.zeros((n_states, n_states))
        rep[nonterminal_idx] = rep_input
    else:
        rep = rep_input

    pi = policy_improvement(rep, w, P, R)

    # reset env
    s = env.reset()

    mse_list = [((rep @ w - V_optimal) ** 2).mean()]
    return_list = [eval_policy(env_eval, pi, n_episodes=n_eval_episodes)]

    pbar = tqdm(total=max_iter)
    for n in range(max_iter):
        # action selection (eps greedy)
        if np.random.rand() < eps:
            a = np.random.choice(n_actions)
        else:
            a = np.random.choice(n_actions, p=pi[s['state']])

        ns, r, done, d = env.step(a)
        
        ### update weight
        # value of curr state
        s_feature = rep[s['state']]
        v = s_feature @ w

        # value of next state
        if d['terminated']:
            v_next = 0
        else:
            v_next = rep[ns['state']] @ w

        w += lr * (r + FLAGS.gamma * v_next - v) * s_feature

        # plot_value_pred_map(env, w, contain_goal_value=True)
        # plt.show()

        # plot_value_pred_map(env, rep @ w, True)
        # plt.show()
        # quit()

        ### policy improvement
        pi = policy_improvement(rep, w, P, R)

        if done:
            s = env.reset()
        else:
            s = ns


        if (n + 1) % log_interval == 0:
            V = rep @ w
            # print(V[-1])
            mse = ((V - V_optimal) ** 2).mean()
            mse_list.append(mse)

            policy_value = eval_policy(env_eval, pi, n_episodes=n_eval_episodes)
            return_list.append(policy_value)

            pbar.set_description(f"MSE: {mse}; Return: {policy_value}")
            pbar.update(log_interval)

            # print(V[109])

            # print(f"{n + 1}:", mse, policy_value)

    return w, mse_list, return_list

def policy_evaluation(rep, weight, pi, P, R, max_iter=10000):
    """
    rep: representation (SR, DR, or MER), (S, S)
    weight: weights for value prediction, (S, )
    pi: current policy (S, A)
    P: transition prob matrix (S, A, S)
    R: reward vector, (S)
    """
    P_pi = (P * pi[..., None]).sum(1)
    n_iter = 0
    while True:

        weight_old = weight.copy()

        value_pred = rep @ weight

        weight += FLAGS.VI_step_size * rep.T @ (R + FLAGS.gamma * P_pi @ value_pred - value_pred)

        n_iter += 1

        if ((weight_old - weight) ** 2).mean() < 1e-5:
            break

        if n_iter >= max_iter:
            break

    return weight

def policy_improvement(rep, weight, P, R):
    """
    rep: representation (SR, DR, or MER), (S, S)
    weight: weights for value prediction, (S, )
    P: transition prob matrix (S, A, S)
    R: reward vector, (S)
    """

    value_pred = rep @ weight
    Q = P @ value_pred
    pi_new = (Q == Q.max(1, keepdims=True)).astype(float)
    pi_new /= pi_new.sum(1, keepdims=True)
    
    return pi_new

def policy_iteration(rep, weight_init, pi_init, P, R):
    """
    rep: representation (SR, DR, or MER), (S, S)
    weight: weights for value prediction, (S, )
    pi: current policy (S, A)
    P: transition prob matrix (S, A, S)
    R: reward vector, (S)
    """
    weight = weight_init.copy()
    pi = pi_init.copy() 

    while True:
        pi_old = pi.copy()

        weight = policy_evaluation(rep, weight, pi, P, R)
        pi = policy_improvement(rep, weight, P, R)

        if np.allclose(pi_old, pi):
            break

    return weight, pi


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

def plot_fit_weights_result(env, lrs, plot_each):
    """
    Plot results of experiment of fitting weights to estimate optimal V directly.

    plot_each: 
        If True, plot results of all learning rates for each representation.
        If False, plot best result (best lr) for each representation.
            Determine best by looking at last mse.
    """

    path = join("minigrid_basics", "examples", "rep_plots", "rep_fit", env.unwrapped.spec.id)

    reps = ["SR", "DR", "MER"]
    colors = ["red", "green", "blue", "magenta", "cyan", "black", "brown", "orange", "yellow"]
    markers = ['o', 's', '^', 'h']
    x = None

    best_w_for_reps = []    # keep track of best learned weights for each representation

    for i, rep in enumerate(reps):
    
        rep_path = join(path, rep)

        fnames = os.listdir(rep_path)
        fnames = [d for d in fnames if "pkl" in d]    # filter non data files

        lrs = [d.split(".pkl")[0] for d in fnames]       # strings
        
        mse_list = []
        w_list = []
        for j, (lr, fname) in enumerate(zip(lrs, fnames)):
            with open(join(rep_path, fname), "rb") as f:
                data = pickle.load(f)
                mse = data['mse']
                mse_list.append(mse)
                w_list.append(data['w_fit'])

            if x is None:
                x = np.array(range(len(mse))) * 100

            if plot_each:
                plt.plot(x, mse, label=lr, color=colors[j])

        if plot_each:
            plt.ylim([0, 200])
            plt.title(rep)
            plt.legend()
            plt.show()
            continue

        mse_list = np.array(mse_list)
        best_idx = np.nanargmin(mse_list[:, -1])    # ignore nans

        plt.plot(x, mse_list[best_idx], label=rep, color=colors[i])   

        best_w_for_reps.append(w_list[best_idx])     

    if not plot_each:
        plt.ylim([0, 100])
        plt.legend()
        plt.xlabel("Number of iterations")
        plt.ylabel("MSE between Predicted and True V")
        plt.title(env.unwrapped.spec.id)
        plt.show()

    return best_w_for_reps
    
def plot_learned_V_given_weights(env, w):
    """
    w: list of best weights for each representation
    """
    assert len(w) == 3      # for SR, DR, MER
    path = join("minigrid_basics", "examples", "rep_plots", "rep_fit", env.unwrapped.spec.id)

    n_states = env.num_states
    n_actions = env.num_actions
    P = env.transition_probs
    R = env.rewards
    nonterminal_idx = (P.sum(-1).sum(-1) != 0)
    pi_uniform = np.ones((n_states, n_actions)) / n_actions

    # get optimal V
    V_optimal = value_iteration(P, R)
    V_optimal = V_optimal[nonterminal_idx]

    # get best V learned using SR
    SR = compute_SR(pi_uniform, P, FLAGS.gamma)
    SR_processed = SR[nonterminal_idx]
    V_SR = SR_processed @ w[0]

    # SR = np.zeros((n_states, n_states))
    # SR[nonterminal_idx] = SR_processed
    # pi = policy_improvement(SR, w[0], P, R)
    # print(eval_policy(env, pi, 1))
    # quit()
    
    
    # DR
    DR = compute_DR(pi_uniform, P, R)
    DR_processed = process_DR_or_MER(DR[nonterminal_idx])
    V_DR = DR_processed @ w[1]

    # plot_value_pred_map(env, value_prediction=DR_processed[-1] * w[1], contain_goal_value=True)
    # print((DR_processed[-1] * w[1]).sum())
    # plt.show()
    # quit()

    # DR = np.zeros((n_states, n_states))
    # DR[nonterminal_idx] = DR_processed
    # pi = policy_improvement(DR, w[0], P, R)
    # print(eval_policy(env, pi, 1))
    # quit()
    
    # MER
    MER = compute_MER(P, R, n_states, n_actions)
    MER_processed = process_DR_or_MER(MER[nonterminal_idx])
    V_MER = MER_processed @ w[2]

    # MER = np.zeros((n_states, n_states))
    # MER[nonterminal_idx] = MER_processed
    # pi = policy_improvement(MER, w[0], P, R)
    # print(eval_policy(env, pi, 1))
    # quit()

    ### plot
    vmin = np.min([V_optimal, V_SR, V_DR, V_MER])
    vmax = np.max([V_optimal, V_SR, V_DR, V_MER])
    v_range = (vmin, vmax)

    # plot the learned value functions
    num_rows = 1
    plt.figure(figsize=(15, 5))
    plt.subplot(num_rows, 4, 1)
    plot_value_pred_map(env, V_optimal, v_range=v_range)
    plt.title("Optimal V")

    plt.subplot(num_rows, 4, 2)
    plot_value_pred_map(env, V_SR, v_range=v_range)
    plt.title("V SR")

    plt.subplot(num_rows, 4, 3)
    plot_value_pred_map(env, V_DR, v_range=v_range)
    plt.title("V DR")

    plt.subplot(num_rows, 4, 4)
    plot_value_pred_map(env, V_MER, v_range=v_range)
    plt.title("V MER")
    # plt.colorbar()

    plt.tight_layout()
    plt.savefig(join(path, "learned_V.png"))
    plt.show()


    # plot diff between learned and optimal value funnction
    mse_SR = V_SR - V_optimal
    mse_DR = V_DR - V_optimal
    mse_MER = V_MER - V_optimal

    vmin = np.min([mse_SR, mse_DR, mse_MER])
    vmax = np.max([mse_SR, mse_DR, mse_MER])
    v_range = (vmin, vmax)

    plt.figure(figsize=(15, 5))
    plt.subplot(num_rows, 3, 1)
    plot_value_pred_map(env, mse_SR, v_range=v_range)
    plt.title("V SR - V Opt")
    
    plt.subplot(num_rows, 3, 2)
    plot_value_pred_map(env, mse_DR, v_range=v_range)
    plt.title("V DR - V Opt")

    plt.subplot(num_rows, 3, 3)
    plot_value_pred_map(env, mse_MER, v_range=v_range)
    plt.title("V MER - V Opt")
    # plt.colorbar()

    plt.tight_layout()
    plt.savefig(join(path, "V_diff.png"))
    plt.show()

    # plot learned weights
    vmin = np.min(w)
    vmax = np.max(w)
    v_range = (vmin, vmax)

    plt.figure(figsize=(15, 5))
    plt.subplot(num_rows, 3, 1)
    plot_value_pred_map(env, w[0], v_range=v_range, contain_goal_value=True)
    plt.title("w SR")

    plt.subplot(num_rows, 3, 2)
    plot_value_pred_map(env, w[1], v_range=v_range, contain_goal_value=True)
    plt.title("w DR")

    plt.subplot(num_rows, 3, 3)
    plot_value_pred_map(env, w[2], v_range=v_range, contain_goal_value=True)
    plt.title("w MER")
    # plt.colorbar()

    plt.tight_layout()
    plt.savefig(join(path, "learned_w.png"))
    plt.show()

def fit_weight(env, rep, true_value, lr, P, R, max_iter=15000):
    """
    Fix weight for true value function using representation.

    lr: learning rate
    """
    n_states = env.num_states
    weight = np.zeros((rep.shape[1]))
    mse_list = [((true_value - rep @ weight ) ** 2).mean()]
    n_iter = 0

    # nonterminal_idx = (P.sum(-1).sum(-1) != 0)
    # rep_full = np.zeros((n_states, n_states))
    # rep_full[nonterminal_idx] = rep

    pbar = tqdm(total=max_iter)
    for n in range(max_iter):
        # lr = 1e-3 / (n + 1)
        weight += lr * rep.T @ (true_value - rep @ weight)
        n_iter += 1

        if n_iter % 100 == 0:
            mse = ((true_value - rep @ weight ) ** 2).mean()
            pbar.set_description(f"MSE: {mse}")
            pbar.update(100)
            mse_list.append(mse)

            # pi = policy_improvement(rep_full, weight, P, R)
            # ret = eval_policy(env, pi, 1)
            # print(ret)
            

    return weight, mse_list

def fit_weights_using_reps(env, rep, lrs, verbose=True):
    """
    Compute the optimal V, then use as target to fit
    weights for SR, DR, MER.

    Params:
    env: environment
    lrs: list of learning rates to use
    """

    if rep not in ["SR", "DR", "MER"]:
        raise ValueError(f"Representation {rep} not recognized.")

    path = join("minigrid_basics", "examples", "rep_plots", "rep_fit", env.unwrapped.spec.id)
    os.makedirs(path, exist_ok=True)
    
    R = env.rewards
    P = env.transition_probs
    nonterminal_idx = (P.sum(-1).sum(-1) != 0)  #idx of non-terminal states

    n_states = env.num_states
    n_actions = env.num_actions

    V_optimal = value_iteration(P, R)
    V_optimal = V_optimal[nonterminal_idx]  # only focus on learning for non-terminal states, because V(terminal) = 0

    # compute wrt to uniform random policy
    pi_uniform = np.ones((n_states, n_actions)) / n_actions

    if rep == "SR":
        SR = compute_SR(pi_uniform, P, FLAGS.gamma)
        SR_processed = SR[nonterminal_idx]
        M = SR_processed
    elif rep == "DR":
        DR = compute_DR(pi_uniform, P, R)
        DR_processed = process_DR_or_MER(DR[nonterminal_idx])
        M = DR_processed
    elif rep == "MER":
        MER = compute_MER(P, R, n_states, n_actions)
        MER_processed = process_DR_or_MER(MER[nonterminal_idx])
        M = MER_processed

    if verbose:
        print(">> Fitting using", rep)

    # start fitting
    for lr in lrs:
        # there's no stochasticity in just fitting the optimal V
        # no need to run for multiple seeds

        if verbose:
            print("  Learning rate:", lr)

        w_fit, mse = fit_weight(env, M, V_optimal, lr, P, R)

        save_path = join(path, rep)
        os.makedirs(save_path, exist_ok=True)

        data_dict = dict(
            w_fit = w_fit,
            mse = mse
        )

        with open(join(save_path, f"{lr}.pkl"), "wb") as f:
            pickle.dump(data_dict, f)


def eigen_approx_V(env, rep, rep_name):
    """
    Use eigenvectors of representations to approximate V optimal.
    """
    R = env.rewards
    P = env.transition_probs
    V_optimal = value_iteration(P, R)
    # nonterminal_idx = (P.sum(-1).sum(-1) != 0)
    # V_optimal = V_optimal[nonterminal_idx]

    eigenvalues, eigenvectors = np.linalg.eig((rep + rep.T) / 2)
    eigenvectors = np.real(eigenvectors)

    # sort eigenvectors
    idx = eigenvalues.argsort()
    eigenvectors = eigenvectors.T[idx[::-1]].T

    # print((eigenvectors.T @ eigenvectors).sum(0))
    print(np.linalg.matrix_rank(eigenvectors.T @ eigenvectors))

    # coefficients = np.linalg.inv(eigenvectors.T @ eigenvectors) @ eigenvectors.T @ V_optimal
    
    def mse(V_optimal, V_approx):
        return ((V_optimal - V_approx) ** 2).mean()

    mse_list = []
    for i in range(50): #range(P.shape[0]):
        try:
            vecs = eigenvectors[:, :i + 1]

            coeff = np.linalg.inv(vecs.T @ vecs) @ vecs.T @ V_optimal

            V_approx = vecs @ coeff
            mse_list.append(mse(V_optimal, V_approx))
        except:
            print(i)
            quit()

    # for i in range(50): #range(len(coefficients)):
    #     c = coefficients[:i + 1]

    #     V_approx = eigenvectors[:, :i + 1] @ c
    #     mse_list.append(mse(V_optimal, V_approx))

    plt.plot(np.array(mse_list) + 1, label=rep_name)


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

    if FLAGS.fit_optimal_V:
        path = join("minigrid_basics", "examples", "rep_plots", "rep_fit", env.unwrapped.spec.id)
    elif FLAGS.learn_V_TD:
        path = join("minigrid_basics", "examples", "rep_plots", "rep_learn_td", env.unwrapped.spec.id)

    env_eval = gym.make(env_id)
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)

    R = env.rewards
    P = env.transition_probs
    nonterminal_idx = (P.sum(-1).sum(-1) != 0)  #idx of non-terminal states
    n_nonterminal_states = nonterminal_idx.astype(int).sum()

    n_states = env.num_states
    n_actions = env.num_actions
    # print("Number of states:", n_states)
    # print("Number of actions", n_actions)

    ##############################
    ### Compute optimal V
    ##############################

    V_optimal = value_iteration(P, R)
    V_optimal = V_optimal[nonterminal_idx]
    plot_value_pred_map(env, V_optimal)
    plt.colorbar()
    plt.title("Optimal V")
    if FLAGS.plot:
        plt.savefig(join(path, "reward_map.png"))
    plt.clf()


    ##############################
    ### Compute & Visualize Reps
    ##############################

    # ### compute SR wrt to uniform random policy
    # pi_uniform = np.ones((n_states, n_actions)) / n_actions
    # SR = compute_SR(pi_uniform, P, FLAGS.gamma)
    # SR_processed = SR[nonterminal_idx][:, nonterminal_idx]

    # w_opt_SR = np.linalg.inv(SR_processed) @ V_optimal

    # V_opt_SR = SR_processed @ w_opt_SR
    # assert np.allclose(V_optimal, V_opt_SR)
    # plt.subplot(1, 4, 2)
    # plot_value_pred_map(env, V_opt_SR)
    # plt.title("")

    # ### Compute DR
    # DR = compute_DR(pi_uniform, P, R)
    # DR_processed = process_DR_or_MER(DR[nonterminal_idx][:, nonterminal_idx])

    # w_opt_DR = np.linalg.inv(DR_processed) @ V_optimal
    # V_opt_DR = DR_processed @ w_opt_DR
    # assert np.allclose(V_opt_DR, V_optimal)
    # plt.subplot(1, 4, 3)
    # plot_value_pred_map(env, V_opt_DR)

    # ### Compute MER
    # MER = compute_MER(P, R, n_states, n_actions)
    # MER_processed = process_DR_or_MER(MER[nonterminal_idx][:, nonterminal_idx])


    # w_opt_MER = np.linalg.inv(MER_processed) @ V_optimal
    # V_opt_MER = MER_processed @ w_opt_MER
    # assert np.allclose(V_optimal, V_opt_MER)
    # plt.subplot(1, 4, 4)
    # plot_value_pred_map(env, V_opt_MER)


    ##############################
    ### Fit weights using Reps
    ##############################

    if FLAGS.fit_optimal_V:
        lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        lrs = [1]
        if FLAGS.plot:
            best_weights = plot_fit_weights_result(env, lrs, FLAGS.plot_each)
            if not FLAGS.plot_each:
                plot_learned_V_given_weights(env, best_weights)
        else:
            fit_weights_using_reps(env, "DR", lrs)


    pi_uniform = np.ones((n_states, n_actions)) / n_actions
    # SR = compute_SR(pi_uniform, P, FLAGS.gamma)
    # SR_processed = SR[nonterminal_idx][:, nonterminal_idx]
    # # w_SR, _ = fit_weight(SR_processed, V_optimal, 2e-4)

    # DR = compute_DR(pi_uniform, P, R)
    # DR_processed = process_DR_or_MER(DR[nonterminal_idx][:, nonterminal_idx])
    # # w_DR, _ = fit_weight(DR_processed, V_optimal, 9e-4)

    # MER = compute_MER(P, R, n_states, n_actions)
    # MER_processed = process_DR_or_MER(MER[nonterminal_idx][:, nonterminal_idx])
    # # w_MER, _ = fit_weight(MER_processed, V_optimal, 1e-3)


    # for state in range(env.num_states):
    #     plt.subplot(1, 3,  1)
    #     plot_value_pred_map(env, np.abs(SR_processed[state]))

    #     plt.subplot(1, 3,  2)
    #     plot_value_pred_map(env, np.abs( DR_processed[state]))

    #     plt.subplot(1, 3, 3)
    #     plot_value_pred_map(env, np.abs(MER_processed[state]))

    #     plt.show()

    # plt.subplot(1, 3,  1)
    # plot_value_pred_map(env, np.abs(SR_processed).sum(0))
    # plot_value_pred_map(env, np.var(SR_processed, axis=0))


    # plt.subplot(1, 3,  2)
    # plot_value_pred_map(env, np.abs(DR_processed).sum(0))
    # plot_value_pred_map(env, np.var(DR_processed, axis=0))


    # plt.subplot(1, 3, 3)
    # plot_value_pred_map(env, np.abs(MER_processed).sum(0))
    # plot_value_pred_map(env, np.var(MER_processed, axis=0))


    # plt.show()


    # states = [86, 88]
    # n = len(states)
    # for i, state in enumerate(states):
    #     plt.subplot(n + 1, 3, i * 3 + 1)
    #     plot_value_pred_map(env, SR_processed[state])

    #     plt.subplot(n + 1, 3, i * 3 + 2)
    #     plot_value_pred_map(env, DR_processed[state])

    #     plt.subplot(n + 1, 3, i * 3 + 3)
    #     plot_value_pred_map(env, MER_processed[state])

    # for i, w in enumerate(best_weights):
    #     plt.subplot(n + 1, 3, n * 3 + i + 1)
    #     plot_value_pred_map(env, w)


    # plt.show()

    # def average_feature_variance(M):
    #     vars = []
    #     for row in M:
    #         vars.append(np.var(row))
    #     return np.mean(vars)
    

    # print(average_feature_variance(SR_processed))
    # print(average_feature_variance(DR_processed))
    # print(average_feature_variance(MER_processed))
    
    


    ##############################
    ### TD using Reps
    ##############################

    if FLAGS.learn_V_TD:

        if FLAGS.plot:
            plot_learn_V_TD_result(env)
            quit()

        rep = "MER"

        SR = compute_SR(pi_uniform, P, FLAGS.gamma)
        SR_processed = SR[nonterminal_idx]
        DR = compute_DR(pi_uniform, P, R)
        DR_processed = process_DR_or_MER(DR[nonterminal_idx])
        MER = compute_MER(P, R, n_states, n_actions)
        MER_processed = process_DR_or_MER(MER[nonterminal_idx])
        V_optimal = value_iteration(P, R)

        lrs = [3e-5, 1e-4, 3e-4]#, 1e-3, 3e-3]
        # lrs = [3e-6]

        if rep == "SR":
            M = SR_processed
        elif rep == "DR":
            M = DR_processed
        elif rep == "MER":
            M = MER_processed

        for lr in lrs:
            learn_V_TD_per_lr(env, env_eval, M, rep, P, R, V_optimal, lr, n_seeds=5)


    if FLAGS.eigen_approx_V:
        SR = compute_SR(pi_uniform, P, FLAGS.gamma)
        DR = compute_DR(pi_uniform, P, R)
        MER = compute_MER(P, R, n_states, n_actions)

        for rep, rep_name in zip([SR, DR, MER], ["SR", "DR", "MER"]):
            eigen_approx_V(env, rep, rep_name)
        
        plt.legend()
        plt.xlabel("Number of Largest Eigenvectors")
        plt.ylabel("MSE between approximated and optimal V")
        # plt.ylim([0, 1000])

        plt.title(env.unwrapped.spec.id)
        plt.show()

    quit()

    rep_names = ["SR", "DR", "MER"]
    reps = [SR_processed, DR_processed, MER_processed]
    colors = ["red", "green", "blue"]
    markers = ['o', 's', '^', 'h']

    n_seeds = 5
    for rep, rep_name, c in zip(reps, rep_names, colors):
        print(">>>", rep_name)
        for lr, m in zip([0.0001, 0.0003, 0.001], markers):
            FLAGS.VI_step_size = lr

            mse_list = []
            for seed in range(n_seeds):

                w, mse = TD_eval_model_improv(env, env_eval, rep, P, R, V_optimal)
                mse_list.append(mse)

            mse = np.array(mse_list)

            mse_mean = mse.mean(0)
            mse_std = mse.std(0, ddof=1)
            conf_int = 1.96 / np.sqrt(n_seeds) * mse_std
            x = np.array(range(len(mse_mean))) * 200

            plt.plot(x, mse_mean, label=f"{rep_name}-{lr}", color=c, marker=m, alpha=0.5)
            plt.fill_between(x, mse_mean - conf_int, mse_mean + conf_int, color=c, alpha=0.3)

    plt.legend()
    plt.xlabel("Number of Interactions with Env")
    plt.ylabel("MSE between Predicted & Optimal V")
    plt.show()
    
    





if __name__ == '__main__':
    app.run(main)

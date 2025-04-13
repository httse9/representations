import numpy as np
import matplotlib.pyplot as plt

def compute_SR(env, gamma=0.99):
    """
    Compute SR

    Assume wrt to uniform random policy
    """
    n_states = env.num_states
    n_actions = env.num_actions
    P = env.transition_probs

    # assume uniform random policy
    pi = np.ones((n_states, n_actions)) / n_actions

    # compute P_pi, prob of state transition under pi
    P_pi = (P * pi[..., None]).sum(1)

    SR = np.linalg.inv(np.eye(n_states) - gamma * P_pi)
    return SR

def compute_DR(env, lamb=1):
    """
    Compute DR.
    Assume uniform random default policy

    """
    n_states = env.num_states
    n_actions = env.num_actions
    R = env.rewards
    P = env.transition_probs
    pi = np.ones((n_states, n_actions)) / n_actions

    P_pi = (P * pi[..., None]).sum(1)
    DR = np.linalg.inv(np.diag(np.exp(-R / lamb)) - P_pi)
    return DR

def compute_MER(env, lamb = 1):
    """
    Compute MER
    """
    n_states = env.num_states
    n_actions = env.num_actions
    R = env.rewards
    P = env.transition_probs
    pi = np.ones((n_states, n_actions)) / n_actions

    P_pi = (P * pi[..., None]).sum(1)
    MER = np.linalg.inv(np.diag(np.exp(-R / lamb)) / n_actions - P_pi)
    return MER

def get_representation(env, rep_name, gamma=0.99, eigen=False):
    """
    Compute representation

    eigen: if true, return eigenvector of representation instead of rep.
    """
    if rep_name == "SR":
        rep = compute_SR(env, gamma=gamma)
    elif rep_name == "DR":
        rep = compute_DR(env)
    elif rep_name == "MER":
        rep = compute_MER(env)
    else:
        raise ValueError(f"Representation {rep_name} not recognized.")
    
    if eigen:
        # return eigenvectors as rows
        v =  np.linalg.eig((rep + rep.T) / 2)[1].T
        return np.real(v)
    else:
        return rep

def process_DR_or_MER(M):
    """
    Since DR and MER values are too small and close, 
    process them before use.
    * Assumes that only contain rows of non-temrinal states.
    After processing, entries dist is roughly normal.

    M: matrix to be processed
    """
    assert (M > 0).all()

    M_processed = M.copy()
    # apply log
    M_processed = np.log(M_processed)
    # normalize
    M_processed /= np.abs(M_processed).max()

    return M_processed

def value_iteration(env, gamma=0.99):
    """
    VI to get ground truth value function
    P: transition prob matrix (S, A, S)
    R: reward vector, (S)
    """
    R = env.rewards
    P = env.transition_probs

    value = np.zeros((env.num_states))

    while True:
        value_old = value.copy()

        value = R + gamma * (P @ value).max(1)

        if ((value_old - value) ** 2).sum() < 1e-10:
            break

    return value

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

def plot_mean_and_conf_interval(x, y, label, color, alpha):

    """
    x: x axis
    y: data, each row contains one trial
    """
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    n_trials = np.array(y).shape[0]

    y_interval = 1.96 / np.sqrt(n_trials) * y_std

    plt.plot(x, y_mean, label=label, color=color, alpha=alpha)
    plt.fill_between(x, y_mean - y_interval, y_mean + y_interval, alpha=0.05, color=color)

def construct_value_pred_map(env, value_prediction, contain_goal_value=False):
    """
    Take the vector of predicted values, and visualize in the environment map.

    Params:
    value_prediction: vector of predicted state values, shape (S)
    contain_goal_value: whether the input value_prediction contains the value \ 
        prediction of goal values. If False, 
    """
    state_num = 0
    value_pred_map = np.zeros(env.reward_grid.T.shape) - float('-inf')

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
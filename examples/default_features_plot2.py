import numpy as np
import os
import gin
import gym
import matplotlib.pyplot as plt
from itertools import product
import types
import argparse
import random
import pickle
from os.path import join
from minigrid_basics.examples.plotter import Plotter


# testing imports
import gin
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.default_features import QLearner, SuccessorFeatureLearner, DefaultFeatureLearnerSA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_test_data():

    path = join("minigrid_basics", "experiments", "transfer", env_name)

    with open(join(path, "test_data.pkl"), "rb") as f:
        data = pickle.load(f)

    return data


def read_SF():
    path = join("minigrid_basics", "experiments", "transfer", env_name, "SF")
    SFs = []

    for seed in range(1, 51):
        with open(join(path, f"{seed}.pkl"), "rb") as f:
            data = pickle.load(f)

        SFs.append(data)
    return SFs

def read_DF():
    path = join("minigrid_basics", "experiments", "transfer", env_name, "DF")
    DFs = []
    for seed in range(1, 51):
        with open(join(path, f"{seed}.pkl"), "rb") as f:
            data = pickle.load(f)

        DFs.append(data)
    return DFs

def compute_SF_transfer_policy(terminal_reward, SF):
    """
    SF: SF for diff policies for 1 seed
    """
    sf_w = np.array([-1, -20, 0, 0, 0, 0])
    sf_w[2:] = terminal_reward

    # GPI
    sf_pis = []
    Qs_8 = [sf @ sf_w for sf in SF]


    for num_policy in [1, 2, 4, 8]:
        Qs = [q.reshape(env.num_states, 1, env.num_actions) for q in Qs_8[:num_policy]]
        Qs = np.concatenate(Qs, axis=1)
        Qs = Qs.max(1)
        sf_pi = Qs.argmax(1)

        sf_pis.append(sf_pi)

    return sf_pis

def compute_DF_transfer_policy(terminal_reward, DF):
    terminal_reward = [r for r in terminal_reward for _ in range(env.num_actions)]
    Q = DF @ np.exp(terminal_reward)
    Q = Q.reshape(-1, 4)
    pi = Q.argmax(1)

    return pi

def eval_policy(env, terminal_reward, pi):
    ret = 0
    s = env.reset()
    done = False
    while not done:
        a = pi[s['state']]
        ns, r, done, d = env.step(a)
        ret += r
        s = ns

    if d['terminated']:
        idx = env.terminal_idx.index(s['state'])
        ret += terminal_reward[idx]

    return ret


if __name__ == "__main__":
    env_name = "fourrooms_multigoal"
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=42, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    np.random.seed(42)
    random.seed(42)

    test_data = read_test_data()
    SFs = read_SF()
    DFs = read_DF()

    opt_returns = []
    SF_returns = []
    DF_returns = []

    fontsize=12
    plt.rcParams.update({
    'font.size': fontsize  # set your preferred default size here
    })

    visualizer = Visualizer(env)
    plt.figure(figsize=(12, 15))
    
    plotter = Plotter()

    terminal_rewards = [
        [100, 0, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 100],
        [100, 0, 0, 100],
        [0, 100, 100, 0],
        [-100, 0, 0, 0],
        [100, 80, 80, 100]
    ]

    # for terminal_reward, pi_opt in zip(test_data['terminal_rewards'], test_data['pis']):

    plt.subplot(len(terminal_rewards), 5, 1)

    for i, terminal_reward in enumerate(terminal_rewards):
    
        # pi_opt = pi_opt.argmax(1)
        # print(terminal_reward)

        sf_pis = [compute_SF_transfer_policy(terminal_reward, SF) for SF in SFs]    # SF for diff #policies, [50, 4, ...]
        sf_pis = np.array(sf_pis).transpose(1, 0, 2)    # [4, 50, num_states]

        df_pis = [compute_DF_transfer_policy(terminal_reward, DF) for DF in DFs]
        df_pis = np.array(df_pis)

        # print(sf_pis.shape)
        # print(df_pis.shape)


        # print(df_pis[0:1].shape)
        # print(sf_pis[:, 0].shape)

        # DF, SF-1, SF-2, SF-4, SF-8
        policies = []
        policies.append(df_pis[np.random.randint(0, 50)].reshape(1, -1))
        for pi in sf_pis:
            policies.append(pi[np.random.randint(0, 50)].reshape(1, -1))
        policies = np.concatenate(policies, axis=0)

        for j, pi in enumerate(policies):
            plt.subplot(len(terminal_rewards), 5, i * 5 + j + 1)

            policy = {}
            policy['policy'] = pi
            policy['termination'] = np.zeros((pi.shape[0]))
            for idx in env.terminal_idx:
                policy['termination'][idx] = 1.

            visualizer.visualize_option(policy)

    plt.tight_layout()
    plt.savefig("minigrid_basics/plots/transfer_Figure_2.png", dpi=300)
    plt.show()

       


    


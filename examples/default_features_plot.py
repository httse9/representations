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
    plt.figure(figsize=(7, 3.5))
    plt.subplot(1, 2, 1)
    visualizer.visualize_env()

    plt.subplot(1, 2, 2)
    plotter = Plotter()

    for terminal_reward, pi_opt in zip(test_data['terminal_rewards'], test_data['pis']):
        pi_opt = pi_opt.argmax(1)
        # print(terminal_reward)

        sf_pis = [compute_SF_transfer_policy(terminal_reward, SF) for SF in SFs]    # SF for diff #policies, [50, 4, ...]
        sf_pis = np.array(sf_pis).transpose(1, 0, 2)    # [4, 50, num_states]

        df_pis = [compute_DF_transfer_policy(terminal_reward, DF) for DF in DFs]

        opt_return = []
        SF_ret = []
        DF_ret = []
        for i in range(1): # generate 1 random start states

            # generate random start state and fix
            s = env.reset()
            x, y = env.state_to_pos[s['state']]
            env.raw_grid[x, y] = 's'

            ### eval policies
            # optimal policy
            opt_r = eval_policy(env, terminal_reward, pi_opt)
            opt_return.append(opt_r)

            # SF policies
            SF_return = []
            for j, num_policies in enumerate([1, 2, 4, 8]):
                sf_returns = [eval_policy(env, terminal_reward, pi) for pi in sf_pis[j]]
                SF_return.append(sf_returns)
            SF_return = np.array(SF_return)
            SF_ret.append(SF_return)

            df_returns = [eval_policy(env, terminal_reward, pi) for pi in df_pis]
            DF_ret.append(df_returns)

            # print("  ", opt_return, np.mean(sf_returns), np.mean(df_returns))

            # unfix start state
            env.raw_grid[x, y] = ' '

        opt_return = np.mean(opt_return, 0)
        SF_ret = np.mean(SF_ret, 0)
        DF_ret = np.mean(DF_ret, 0)

        opt_returns.append(opt_return)
        SF_returns.append(SF_ret)
        DF_returns.append(DF_ret)

    opt_returns = np.array(opt_returns)
    DF_returns = np.array(DF_returns).transpose(1, 0)

    for optr, dfr in zip(opt_returns, DF_returns.T):
        if (optr < dfr).any():
            idx = np.where(optr < dfr)
            print(optr, dfr[idx])


    opt_returns = np.cumsum(opt_returns, -1)    # [100,]

    
    DF_returns = np.cumsum(DF_returns, -1)      # [50, 100]
    # print(DF_returns.shape)

    DF_returns_mean = DF_returns.mean(0)
    idx = (opt_returns < DF_returns_mean)
    # print(DF_returns_mean[idx])
    # print(opt_returns[idx])

    SF_returns = np.array(SF_returns).transpose(1, 2, 0)    # [4, 50, 100]
    SF_returns = np.cumsum(SF_returns, -1)
    # print(SF_returns.shape)

   
    ax = plt.gca()
    x = range(len(opt_returns))
    ax.plot(opt_returns, color='k', linestyle="--")

    plotter.plot_data(ax, x, DF_returns,)


    for SF_return, num_policy in zip(SF_returns, [1,2,4,8]):
        plotter.index += 1
        plotter.plot_data(ax, x, SF_return)

    ax.text(45, 3100, "Optimal", size=fontsize, color="k",)
    plotter.index = 0
    plotter.draw_text(ax, 45, 2300, "DF", size=fontsize)
    plotter.index += 1
    plotter.draw_text(ax, 60, -900, "SF-1", size=fontsize)
    plotter.index += 1
    plotter.draw_text(ax, 60, 100, "SF-2", size=fontsize)
    plotter.index += 1
    plotter.draw_text(ax, 60, 1250, "SF-4", size=fontsize)
    plotter.index += 1
    plotter.draw_text(ax, 90, 2910, "SF-8", size=fontsize)

    ax.set_yticks([-1000, 0, 1000, 2000, 3000, 4000], [-1,0,1,2,3,4])


    plotter.finalize_plot(ax, xlabel="Terminal Reward Configuration", ylabel="Cumulative Return ($×10^3$)")

    plt.savefig("minigrid_basics/plots/transfer_Figure_1_new.png", dpi=300)
    plt.show()


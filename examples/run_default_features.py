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

# testing imports
import gin
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.default_features import QLearner, SuccessorFeatureLearner, DefaultFeatureLearnerSA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def sample_terminal_reward():
    return np.random.normal(loc=0., scale=50, size=(4,))

def learn_SF(env):
    SFs = []    # successor features for different reward functions

    # how many reward functions to compute SF for
    for i in range(8):
        print("Learning SF for ", i)
        # learn optimal policy
        terminal_reward = sample_terminal_reward()
        qlearner = QLearner(env, 1.0, terminal_reward=terminal_reward)
        qlearner.learn(100000)
        pi = qlearner.get_current_policy()

        sf_learner = SuccessorFeatureLearner(env)
        sf_learner.learn(pi, 100000)

        SFs.append(sf_learner.sf)

    return SFs

def learn_DF(env):
    df_learner = DefaultFeatureLearnerSA(env)
    df_learner.learn(100000)
    return df_learner.df

def generate_test_reward_and_policy(env):
    terminal_rewards = []
    pis = []
    for i in range(100):
        print(i)
        terminal_reward = sample_terminal_reward()
        terminal_rewards.append(terminal_reward)

        qlearner = QLearner(env, 1.0, terminal_reward=terminal_reward)
        qlearner.learn(100000)
        pi = qlearner.get_current_policy()
        pis.append(pi)

    path = join("minigrid_basics", "experiments", "transfer", env_name)

    data = dict(
        terminal_rewards=terminal_rewards,
        pis=pis
    )

    with open(join(path, "test_data.pkl"), "wb") as f:
        pickle.dump(data, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, help="SF or DF")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--generate_test_data", action="store_true")
    args = parser.parse_args()


    env_name = "fourrooms_multigoal"
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=args.seed, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.generate_test_data:
        generate_test_reward_and_policy(env)
        quit()

    # if args.feature == "SF":
    SFs = learn_SF(env)

    path = join("minigrid_basics", "experiments", "transfer", env_name, "SF")
    os.makedirs(path, exist_ok=True)
    with open(join(path, f"{args.seed}.pkl"), "wb") as f:
        pickle.dump(SFs, f)

    # for i in range(10):
    #     sf_w = np.array([-1, -20, 0, 0, 0, 0])
    #     terminal_reward = sample_terminal_reward()
    #     sf_w[2:] = terminal_reward
    #     print(terminal_reward)

    #     # GPI
    #     Qs = [sf @ sf_w for sf in SFs]
    #     Qs = [q.reshape(env.num_states, 1, env.num_actions) for q in Qs]
    #     Qs = np.concatenate(Qs, axis=1)
    #     Qs = Qs.max(1)
    #     sf_pi = Qs.argmax(1)

    #     visualizer = Visualizer(env)
    #     policy = {}
    #     policy['policy'] = sf_pi
    #     policy['termination'] = np.zeros((env.num_states))
    #     for idx in env.terminal_idx:
    #         policy['termination'][idx] = 1.

    #     visualizer.visualize_option(policy)
    #     plt.show()


    # elif args.feature == "DF":
    DF = learn_DF(env)

    path = join("minigrid_basics", "experiments", "transfer", env_name, "DF")
    os.makedirs(path, exist_ok=True)
    with open(join(path, f"{args.seed}.pkl"), "wb") as f:
        pickle.dump(DF, f)

        # for i in range(10):
        #     terminal_reward = sample_terminal_reward()
        #     print(terminal_reward)
        #     terminal_reward = [r for r in terminal_reward for _ in range(env.num_actions)]

        #     Q = DF @ np.exp(terminal_reward)
        #     Q = Q.reshape(-1, 4)
        #     pi = Q.argmax(1)

        #     visualizer = Visualizer(env)
        #     policy = {}
        #     policy['policy'] = pi
        #     policy['termination'] = np.zeros((Q.shape[0]))
        #     for idx in env.terminal_idx:
        #         policy['termination'][idx] = 1.

        #     visualizer.visualize_option(policy)
        #     plt.show()

    # else:

    #     raise ValueError(f"{args.feature} not recognized.")

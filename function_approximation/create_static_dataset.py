"""
Create a minimal fixed dataset with a uniform initial state distibution,
and a uniform policy.

The dataset size is |S| * |A|
"""

import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def create_dataset(env):

    """
    TODO: Wrong!!!
    1. when initialized at low reward region, reward is wrong (solved)
    2. when initialized at goal, transition is wrong
    """

    dataset = []    # (s, a, r, s', is_goal?)
    
    # enumerate over all states
    for s in range(env.num_states):

        x, y = env.state_to_pos[s]

        # if terminal state, 
        if s in env.terminal_idx:
            for a in range(env.num_actions):
                dataset.append((s, 0, 0, 0, 1.))
            continue

        for a in range(env.num_actions):
            env.reset()
            # set start position to s
            env.unwrapped.agent_pos = [x, y]

            # take action a
            ns, r, done, d = env.step(a)

            # record transition
            dataset.append((s, a, r, ns['state'], 0.))
    return dataset


def onehot(num_states, idx):
    v = np.zeros((num_states)) * 1.
    if idx is None:
        return v
    v[idx] = 1
    return v

        


if __name__ == "__main__":

    env_name = "gridroom_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(1)
    env = gym.make(env_id, seed=0, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    dataset = create_dataset(env)
    print(dataset)

    # onehot transformation
    onehot_dataset = []
    for d in dataset:
        s, a, r, ns, term = d
        onehot_dataset.append([
            onehot(env.num_states, s),
            a, r,
            onehot(env.num_states, ns),
            term
        ])

    with open("minigrid_basics/function_approximation/static_dataset/gridroom_onehot_dataset.pkl", "wb") as f:
        pickle.dump(onehot_dataset, f)
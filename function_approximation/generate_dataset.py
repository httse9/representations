"""
Generate dataset by 
1) running a uniform random policy till episode termination
2) from uniform random initial start states
and 
3) record different types of observation.
    - one-hot
    - coordinates
    - image
For each transition, record
- s, a, r, s', r', terminated (whether s' is terminal state)

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
from os.path import join
from copy import deepcopy

"""
TODO:
1. Figure out a way to record all types of observation at once, so no need to generate separately (done)
2. Agent-Env Loop & record transitions
  - how to do random starts?
"""

def generate_dataset(env, N):
    """
    TODO:
    1. generate N samples in total
    """

    dataset = {
        "onehot": [],
        "coordinates": [],
        "image": []
    }


    count = 0
    while True:

        s = env.reset()     # samples a random start state
        one_hot, coordinates, image = get_observations(env, s)

        done = False
        while not done:

            """
            NOTE:
            We use done here since for large environments, it could be extremely difficult to get to terminal state
            Done also has truncation after some steps.
            I think truncation step is in the env spec files in /reward_env
            """

            a = np.random.choice(env.num_actions)
            ns, r, done, d = env.step(a)
            nr = env.reward()
            terminated = float(ns['state'] in env.terminal_idx)

            one_hot_next, coordinates_next, image_next = get_observations(env, ns)

            dataset['onehot'].append([one_hot, a, r, one_hot_next, nr, terminated])
            dataset['coordinates'].append([coordinates, a, r, coordinates_next, nr, terminated])
            dataset['image'].append([image, a, r, image_next, nr, terminated])

            s = ns
            one_hot, coordinates, image = one_hot_next, coordinates_next, image_next

            count += 1
            if count >= N:
                break

        if count >= N:
            break

    return dataset



def get_observations(env, s):
    """
    s: the state returned by environment, not a singe scalar
    """
    one_hot = onehot(env.num_states, s['state'])

    x, y = env.agent_pos
    coordinates = np.array([x / (env.width - 1), y / (env.height - 1)]) - 0.5

    image = env.custom_rgb()

    return one_hot, coordinates, image



def onehot(num_states, idx):
    v = np.zeros((num_states)) * 1.
    if idx is None:
        return v
    v[idx] = 1
    return v

        


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--num_transitions", default=200_000, type=int, help="Number of transitions to geenrate")
    args = parser.parse_args()


    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(1)
    env = gym.make(env_id, no_goal=False, no_start=True)    # no start: random initial state
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    dataset_path = join("minigrid_basics", "function_approximation", "dataset")
    os.makedirs(dataset_path, exist_ok=True)

    dataset = generate_dataset(env, args.num_transitions)
    with open(join(dataset_path, f"{args.env}_dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)

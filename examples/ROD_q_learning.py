import os
import numpy as np
from minigrid_basics.examples.visualizer import Visualizer
from os.path import join
from collections import deque
from itertools import islice

# testing imports
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import random
import argparse
import pickle


class ROD_Q:
    """
    Baseline
    Just Q-Learning
    """

    def __init__(self, env, init=0, n_steps=100, gamma=0.99, epsilon=0.05, step_size=0.1):

        self.env = env
        self.init = init  # -1000, -100, -10, 0

        self.gamma = gamma
        self.epsilon = epsilon      # 0.01, 0.05, 0.1, 0.15, 0.2
        self.step_size = step_size  # 0.01, 0.03, 0.1, 0.3, 1

        # collect samples
        self.n_steps = n_steps      # number of steps in collect samples phase (episode length)

        self.reset()


    def reset(self):
        
        self.dataset = []
        self.Q_performance = []
        self.learn_Q_policy()       # initialize policy
        

    def collect_samples(self,):
       
        s = self.env.reset()
        for n in range(self.n_steps):
            
            # epsilon greedy
            if np.random.rand() < self.epsilon:
                a = np.random.choice(self.env.num_actions)
            else:
                a = self.pi[s['state']] 

            # take env step
            ns, r, done, d = self.env.step(a)

            self.dataset.append((s['state'], a, r, ns['state']))

            s = ns



    def learn_Q_policy(self,):
        """
        Q-learning with batch data.
        Construct option
        """
        Q = np.zeros((self.env.num_states, self.env.num_actions)) + self.init
        Q[self.env.terminal_idx[0]] = 0     # terminal state Q-value is 0

        for i in range(1000):   # max iteration
            max_delta = 0
            for (s, a, r, ns) in reversed(self.dataset):
                
                if s == self.env.terminal_idx[0]:
                    continue

                # update Q
                delta = r + self.gamma * Q[ns].max() - Q[s, a]
                Q[s, a] += self.step_size *  delta

                # keep track of max change
                max_delta = max(np.abs(delta), max_delta)

            if max_delta < 1e-5:
                break

        self.pi = np.argmax(Q, axis=1)
        return self.pi

    def evaluate_Q_policy(self,):
        
        episode_return = 0
        # environment and policy both deterministic, just one episode is enough
        s = self.env.reset()
        for n in range(self.n_steps):

            a = self.pi[s['state']]
            ns, r, done, d = self.env.step(a)
            episode_return += r

            if done:
                # print("Reached goal", d['terminated'], d['truncated'])
                break
            else:
                s = ns
        print(episode_return)
        self.Q_performance.append(episode_return)


    def rod_cycle(self, n_iterations=100):
        
        for i in range(n_iterations): 

            print(f"  [Iteration: {i + 1}]", end="  ")
            
            # collect samples
            self.collect_samples()

            self.learn_Q_policy()
            self.evaluate_Q_policy()



def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  

def create_env(env_name, seed):
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=seed, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    return env

def get_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the ROD cycle.")

    parser.add_argument("--env", type=str, default="fourrooms", help="Environment name")

    # collect sample step
    parser.add_argument("--n_steps", type=int, default=100, help="Number of steps in each episode")
    parser.add_argument("--init", type=int, default=0, help="Initial value for Q-values")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon for epsilon greedy")
    parser.add_argument("--step_size", type=float, default=0.1, help="Q-learning step size")

    # seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":

    n_iterations = {
        "dayan": 50,
        "dayan_2": 50,
        "fourrooms": 50,      
        "fourrooms_2": 50,
        "gridroom": 120,
        "gridroom_2": 120,
        "gridmaze": 120,
        "gridmaze_2": 120,
        "gridmaze_29": 120,
        "gridroom_25": 120
    }
    
    args = vars(get_args())
    env_name = args.pop('env')
    seed = args.pop('seed')

    args['env'] = create_env(env_name, seed)
    set_random_seed(seed)

    if env_name in ["gridroom_25", "gridmaze_29"]:
        args['n_steps'] = 200

    
    rod_cycle = ROD_Q(**args)
    rod_cycle.rod_cycle(n_iterations=n_iterations[env_name])

    print(rod_cycle.Q_performance)

    ### save results
    path = join("minigrid_basics", "experiments", "rod_q", env_name, "qlearning")
    os.makedirs(path, exist_ok=True)
    
    keys = ["init", "epsilon", "step_size"]
    values = [str(args[key]) for key in keys]
    filename = '-'.join(values) + f"-{seed}.pkl"

    data_dict = dict(
        Q_performance=rod_cycle.Q_performance
    )

    with open(join(path, filename), "wb") as f:
        pickle.dump(data_dict, f)


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


class RODCycle_RW:
    """
    Baseline
    no learning, simple random walk
    """

    def __init__(self, env, n_steps=100):

        self.env = env

        # collect samples
        self.n_steps = n_steps      # number of steps in collect samples phase (episode length)

        self.reset()


    def reset(self):

        # useful statistics to keep track of
        self.cumulative_reward = [0]       # cumulative reward obtained so far
        self.all_iteration_state_visits = []      # keep track of state visit at each ROD cycle iteration
        self.cumulative_state_visits = np.zeros((self.env.num_states))   # cumulative state visits
        self.state_visit_percentage = [0]        # percentage of states visited
        

    def collect_samples(self,):
        state_visits = np.zeros((self.env.num_states))      # keep track of state visits in this iteration

        def update_state_visit_percentage():
            self.state_visit_percentage.append((state_visits + self.cumulative_state_visits > 0).astype(float).mean())

        s = self.env.reset()
        state_visits[s['state']] += 1
        update_state_visit_percentage()

        for n in range(self.n_steps):
            
            # uniform random action
            a = np.random.choice(self.env.num_actions)

            # take env step
            ns, r, done, d = self.env.step(a)

            # keep track of useful stats
            state_visits[ns['state']] += 1  
            self.cumulative_reward.append(r)
            update_state_visit_percentage()

            s = ns
            
        self.all_iteration_state_visits.append(state_visits.copy())
        self.cumulative_state_visits += state_visits


    def rod_cycle(self, n_iterations=100):
        """
        Perform ROD cycle until 
        1) all states are visited, or
        2) number of iterations finished
        """
        print("-----------------------")
        print("ROD Cycle Start")
        print("-----------------------")

        

        for i in range(n_iterations): 

            print(f"  [Iteration: {i + 1}]", end="  ")
            
            # collect samples
            self.collect_samples()

            print(f"State Visit %: {self.state_visit_percentage[-1]:.2f}")


        print("-----------------------")
        print("ROD Cycle End")
        print("-----------------------")


        return self.cumulative_reward, self.state_visit_percentage


def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  

def create_env(env_name, seed):
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=seed, no_goal=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    return env

def get_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the ROD cycle.")

    parser.add_argument("--env", type=str, default="fourrooms", help="Environment name")

    # collect sample step
    parser.add_argument("--n_steps", type=int, default=100, help="Number of steps in each episode")
    
    parser.add_argument("--save_state_visit", action="store_true")

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
        "gridmaze_2": 120
    }
    
    args = get_args()
    env_name = args.env
    seed = args.seed


    env = create_env(env_name, seed)
    set_random_seed(seed)

    
    rod_cycle = RODCycle_RW(env, args.n_steps)

    rewards, visit_percentage = rod_cycle.rod_cycle(n_iterations=n_iterations[env_name])

    ### save results
    path = join("minigrid_basics", "experiments", "rod", env_name)
    os.makedirs(path, exist_ok=True)

    

    if args.save_state_visit:
        data_dict = dict(
            all_iteration_state_visits=rod_cycle.all_iteration_state_visits,
            rewards=rewards,
            visit_percentage=visit_percentage
        )
    else:
        data_dict = dict(
            rewards=rewards,
            visit_percentage=visit_percentage
        )

    with open(join(path, f"random_walk_{seed}.pkl"), "wb") as f:
        pickle.dump(data_dict, f)


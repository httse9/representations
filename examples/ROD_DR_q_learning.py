from minigrid_basics.examples.ROD_DR import RODCycle_DR
import os
import numpy as np
from flint import arb_mat, ctx
from itertools import islice

ctx.dps = 100   # important

# testing imports
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import subprocess
import glob
import pickle


class ROD_DR_Q(RODCycle_DR):

    def __init__(self, env, n_steps=100, p_option=0.05, dataset_size=None, learn_rep_iteration=10, representation_step_size=0.1,
                 gamma=0.99, num_options=None, eigenoption_step_size=0.1, lambd=1.3, plot=True):
        super().__init__(env, n_steps=n_steps, p_option=p_option, dataset_size=dataset_size, learn_rep_iteration=learn_rep_iteration,
            representation_step_size=representation_step_size, gamma=gamma, num_options=num_options, eigenoption_step_size=eigenoption_step_size,
            lambd=lambd, plot=plot)
        
        self.Q_performance = []

    def learn_representation(self):
        """
        DR TD learning
        """
        if self.dataset_size is not None:
            dataset = self.dataset[-self.dataset_size:]
        else:
            dataset = self.dataset

        # do one backward pass through dataset for theoretical guarantee
        for (s, a, r, ns) in reversed(dataset):

            if s == self.env.terminal_idx[0]:
                r = -1

            indicator = np.zeros((self.env.num_states))
            indicator[s] = 1
            self.representation[s] += self.representation_step_size * (np.exp(r / self.lambd) * (indicator + self.representation[ns]) - self.representation[s])

        # remaining iterations, do forward pass
        for _ in range(self.learn_rep_iteration - 1):
            for (s, a, r, ns) in dataset:

                if s == self.env.terminal_idx[0]:
                    r = -1

                indicator = np.zeros((self.env.num_states))
                indicator[s] = 1

                self.representation[s] += self.representation_step_size * (np.exp(r / self.lambd) * (indicator + self.representation[ns]) - self.representation[s])


    def learn_Q_policy(self,):
        """
        Q-learning with batch data.
        Construct option
        """
        Q = np.zeros((self.env.num_states, self.env.num_actions)) - 10000  # pessimistic initialization
        Q[self.env.terminal_idx[0]] = 0     # terminal state Q-value is 0

        while True:   
            max_delta = 0
            for (s, a, r, ns) in reversed(self.dataset):
                

                if s == self.env.terminal_idx[0]:
                    continue

                # update Q
                delta = r + self.gamma * Q[ns].max() - Q[s, a]
                Q[s, a] +=  delta

                # keep track of max change
                max_delta = max(np.abs(delta), max_delta)

            if max_delta < 1e-5:
                break

                
        pi = np.argmax(Q, axis=1)
        return pi

    def evaluate_Q_policy(self,):
        pi = self.learn_Q_policy()

        episode_return = 0
        # environment and policy both deterministic, just one episode is enough
        s = self.env.reset()
        for n in range(self.n_steps):

            a = pi[s['state']]
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

            # extra step: evaluate Q-learning on collected dataset
            self.evaluate_Q_policy()

            # update representation
            self.learn_representation()

            # compute eigenvector
            e0 = self.compute_eigenvector()

            # compute eigenoption
            option = self.compute_eigenoption(e0)

            if self.plot:
                self.visualize_cycle(i, e0, option)

            # append option to set of options
            self.options.append(option)

            print(f"State Visit %: {self.state_visit_percentage[-1]:.2f}")

            # # terminate if visited all states
            # if self.state_visit_percentage[-1] == 1.:
            #     break


        print("-----------------------")
        print("ROD Cycle End")
        print("-----------------------")


        return self.cumulative_reward, self.state_visit_percentage



    

if __name__ == "__main__":
    

    env_name = "gridroom_2"
    # env_name = "fourrooms_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(1)
    # tried smaller rep step size, not reversing dataset
    # increasing episode length does help.
    # I think, limiting the use of eigenoption to once in one episode is useful.
    # what was the problem when using seed 1??

    env = gym.make(env_id, seed=42, no_goal=False, max_steps=200)
    env = maxent_mdp_wrapper.MDPWrapper(env, )


    
    rodc = ROD_DR_Q(env, learn_rep_iteration=1, num_options=1, representation_step_size=0.03, dataset_size=100, p_option=0.1, n_steps=200)
    # rodc = ROD_DR_Q(env, learn_rep_iteration=10, num_options=1, representation_step_size=0.01, dataset_size=100, p_option=0.1, n_steps=200)
    

    rewards, visit_percentage = rodc.rod_cycle(n_iterations=100)

    print(rodc.Q_performance)
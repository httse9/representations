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
import subprocess
import glob


class RODCycle:

    def __init__(self, env, n_steps=100, p_option=0.05, dataset_size=None, learn_rep_iteration=10, representation_step_size=0.1,
                 gamma=0.99, num_options=None, eigenoption_step_size=0.1, plot=True):

        self.env = env
        self.visualizer = Visualizer(env)

        # collect samples
        self.n_steps = n_steps      # number of steps in collect samples phase (episode length)
        self.p_option = p_option       # probability of selecting an option

        # learn representation
        self.dataset_size = max(dataset_size, n_steps)      # number of latest transitions used; default None, meaning keep all transitions
        self.learn_rep_iteration = learn_rep_iteration
        self.representation_step_size = representation_step_size
        self.gamma = gamma

        # eigenoption
        self.num_options = num_options      # default is None, meaning we keep all options from all iterations
        self.eigenoption_step_size = eigenoption_step_size

        # plotting
        self.plot = plot
        self.plot_path = "minigrid_basics/SR_ROD"
        self.dpi = 200

        self.reset()


    def reset(self):
        # options
        self.options = deque(maxlen=self.num_options)       # options computed in ROD cycle
        self.current_option = None      # the current option being followed

        # reprsentation
        self.representation = np.zeros((self.env.num_states, self.env.num_states))

        # dataset
        # self.dataset = deque(maxlen=self.dataset_size)   # dataset of transitions for learning representation
        self.dataset = []

        # useful statistics to keep track of
        self.cumulative_reward = [0]       # cumulative reward obtained so far
        self.all_iteration_state_visits = []      # keep track of state visit at each ROD cycle iteration
        self.cumulative_state_visits = np.zeros((self.env.num_states))   # cumulative state visits
        self.state_visit_percentage = [0]        # percentage of states visited
        

    def collect_samples(self,):
        """
        
        """
        state_visits = np.zeros((self.env.num_states))      # keep track of state visits in this iteration

        def update_state_visit_percentage():
            self.state_visit_percentage.append((state_visits + self.cumulative_state_visits > 0).astype(float).mean())

        s = self.env.reset()
        state_visits[s['state']] += 1
        update_state_visit_percentage()
        self.current_option = None

        for n in range(self.n_steps):
            
            # sample option
            if self.current_option is None and np.random.rand() < self.p_option:
                # sample option with prob p_option if 
                # 1) currently not following option,
                # 2) an option is available

                avail_options = [option for option in self.options if option['initiation'][s['state']]]
                if len(avail_options) >= 1:
                    # print("Option selected.")
                    self.current_option = np.random.choice(avail_options)

            # select action
            if self.current_option is not None:
                # if currently taking option, follow option policy
                a = int(self.current_option['policy'][s['state']])
                # print("Following option")
            else:
                # uniform random
                a = np.random.choice(self.env.num_actions)
                # print("Random action")

            # take env step
            ns, r, done, d = self.env.step(a)

            # keep track of useful stats
            state_visits[ns['state']] += 1  
            self.cumulative_reward.append(r)
            update_state_visit_percentage()

            # keep transition data
            self.dataset.append((s['state'], a, r, ns['state']))
            
            s = ns
            # terminate option if reaches termination set
            if self.current_option is not None and self.current_option['termination'][s['state']]:
                # print("Option terminated.")
                self.current_option = None
            
        self.all_iteration_state_visits.append(state_visits.copy())
        self.cumulative_state_visits += state_visits


    def learn_representation(self, ):
        """
        Update SR
        """
        if self.dataset_size is not None:
            dataset = self.dataset[-self.dataset_size:]
        else:
            dataset = self.dataset

        for _ in range(self.learn_rep_iteration):        
            for (s, a, r, ns) in dataset:

                indicator = np.zeros((self.env.num_states))
                indicator[s] = 1

                self.representation[s] += self.representation_step_size * (indicator + self.gamma * self.representation[ns] - self.representation[s])


    def compute_eigenvector(self,):
        """
        Compute top eigenvector of representation
        """
        rep_sym = (self.representation + self.representation.T) / 2

        eigvalue, eigvec = np.linalg.eig(rep_sym)
        idx = eigvalue.argsort()
        eigvec = eigvec.T[idx[::-1]]
        e0 = np.real(eigvec[0])

        # normalize
        e0 /= np.sqrt(e0 @ e0)
        assert np.isclose(e0 @ e0, 1.0)

        # -1 trick
        if np.sum(e0) >= 0:
            e0 *= -1

        return e0
    
    def compute_eigenoption(self, eigvec, ):
        """
        Q-learning with batch data.
        Construct option
        """
        Q = np.zeros((self.env.num_states, self.env.num_actions))

        while True:   
            max_delta = 0
            for (s, a, r, ns) in self.dataset:

                # eigenpurpose
                r = eigvec[ns] - eigvec[s]

                # update Q
                delta = r + self.gamma * Q[ns].max() - Q[s, a]
                Q[s, a] += self.eigenoption_step_size * delta

                # keep track of max change
                max_delta = max(np.abs(delta), max_delta)

            # break if Q-values converge
            if max_delta < 1e-5:
                break
                
        pi = np.argmax(Q, axis=1)
        termination_set = Q.max(axis=1) <= 0

        option = {
            'policy': pi,
            'termination': termination_set,
            'initiation': ~termination_set
        }
        return option

    def visualize_cycle(self, i, eigenvec, eigenoption):
        """
        i: iteration
        """
        fig, axs = plt.subplots(1, 4, figsize=(21, 4))

        titles = ["Cum. State Visit", "Cur. State Visit", "Eigenvector", "Eigenoption"]

        # cumulative state visitation
        self.visualizer.visualize_shaping_reward_2d(self.cumulative_state_visits, axs[0])

        # current state visitation
        self.visualizer.visualize_shaping_reward_2d(self.all_iteration_state_visits[-1], axs[1])

        # eigenvector
        self.visualizer.visualize_shaping_reward_2d(eigenvec, axs[2])

        # option
        self.visualizer.visualize_option(eigenoption, axs[3])

        for tit, ax in  zip(titles, axs):
            ax.set_title(tit)

        fig.suptitle(f"Iteration {i}")
        plt.tight_layout()
        fig.savefig(join(self.plot_path, f"iteration{i}.png"), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)


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

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(42)

    env = gym.make(env_id, seed=42, no_goal=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    rodc = RODCycle(env, learn_rep_iteration=10)

    rewards, visit_percentage = rodc.rod_cycle(n_iterations=10)

    for i in range(1, len(rewards)):
        rewards[i] += rewards[i - 1]

    # save video
    os.chdir("minigrid_basics/SR_ROD")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'iteration%d.png', '-r', '30','-pix_fmt', 'yuv420p', 
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-y', f'ROD_{env_name}.mp4'
    ])

    for file_name in  glob.glob("*.png"):
        os.remove(file_name)



    # plt.plot(rewards)
    # plt.show()

    # plt.plot(visit_percentage)
    # plt.show()

    # rodc.visualizer.visualize_shaping_reward_2d(rodc.cumulative_state_visits / rodc.cumulative_state_visits.sum())
    # plt.show()



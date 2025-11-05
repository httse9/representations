import numpy as np
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.visualizer import Visualizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import os
import subprocess
import glob
from minigrid_basics.examples.reward_shaper import RewardShaper


class DR_Eigvec_Tabular:

    def __init__(self, env, step_size=0.1, lambd=1.):
        self.env = env
        self.step_size = step_size
        self.lambd = lambd

        shaper = RewardShaper(env)
        self.true_eigvec = shaper.DR_top_log_eigenvector(lambd=lambd)
        

        self.visualizer = Visualizer(env)
        os.makedirs("minigrid_basics/function_approximation/dr_eigvec_tabular_plots", exist_ok=True)

        self.reset()


    def reset(self,):
        # stores state visitation
        self.visitation = np.zeros((self.env.num_states))

        # initializes eigenvector
        # self.eigvec = np.random.normal(size=(self.env.num_states))
        # self.eigvec[self.eigvec < 0] *= -1 
        self.eigvec = np.ones((self.env.num_states))

        # initialize cosine similarity
        self.cosine_similarity = []

        self.plot_visitation_and_eigvec(0)


    def update_eigenvector(self, s, a, r, ns, terminated, nr):

        r /= self.lambd
        nr /= self.lambd

        grad_s = self.eigvec[s] * 2 * np.exp(-r) - self.eigvec[ns]
        grad_ns = - self.eigvec[s]

        print(grad_s)

        self.eigvec[s] -= self.step_size * grad_s
        self.eigvec[ns] -= self.step_size * grad_ns

        if terminated:
            grad_ns = self.eigvec[ns] * 2 * np.exp(-nr)
            self.eigvec[ns] -= self.step_size * grad_ns

    def update_visitation(self, s, a, r, ns, terminated):
        self.visitation[s] += 1
        if terminated:
            self.visitation[ns] += 1

    def compute_cosine_similarity(self, v1, v2):
        return v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def update_cosine_similarity(self):
        """
        Compute cosine similarity between learned eigenvector and
        1) true eigenvector
        2) state visitation
        """

        # log of current eigvec
        eigvec = np.log(self.eigvec)

        cs_true_eigvec = self.compute_cosine_similarity(eigvec, self.true_eigvec)
        cs_visitation = self.compute_cosine_similarity(eigvec, self.visitation)

        self.cosine_similarity.append([cs_true_eigvec, cs_visitation])



    def learn(self, n_episodes=100):
        """
        Interacts with the environment to learn the top eigenvector of the DR

        TODO:
        1. agent-env loop, interact for max #episodes (done)
        2. implement eigenvector learning rule. Handle episode termination (diff terminal state diff reward) (done)
        3. call plotting at the end of each episode
        4. Add visualization of hill climbing policy of eigenvector
        """
        for e in range(n_episodes):

            s = self.env.reset()
            done = False
            myopic = False

            while not done:

                # # uniform random policy as the default policy
                # a = np.random.choice(self.env.num_actions)

                # Follow myopic policy induced by eigvec:
                # being episode with following myopic policy
                # until reaches termination of myopic policy
                if myopic:
                    if self.myopic_policy['termination'][s['state']]:
                        a = np.random.choice(self.env.num_actions)
                        myopic = False
                    else:
                        a = self.myopic_policy['policy'][s['state']]
                else:
                    a = np.random.choice(self.env.num_actions)

                # take action
                ns, r, done, d = env.step(a)
                terminated = d['terminated']
                nr = env.reward()

                # update eigenvector
                self.update_eigenvector(s['state'], a, r, ns['state'], terminated, nr)
                # update visitation
                self.update_visitation(s['state'], a, r, ns['state'], terminated)

                s = ns

                self.update_cosine_similarity()

            self.plot_visitation_and_eigvec(e + 1)
    
    def eigvec_myopic_policy(self):
        """
        Get the myopic (hill-climbing policy) for current eigenvector
        """
        termination = np.zeros((self.env.num_states))
        policy = np.zeros((self.env.num_states))

        for s in range(self.env.num_states):

            # handle unvisited state / terminal state
            if self.visitation[s] == 0 or s in self.env.terminal_idx:
                termination[s] = 1
                continue

            # for visited states:
            pos = self.env.state_to_pos[s]  # (x, y): x-th col, y-th row
            value = self.eigvec[s]  # init value
            myopic_a = -1

            for a, dir_vec in enumerate(np.array([
                [1, 0], # right
                [0, 1], # down
                [-1, 0],    # left
                [0, -1],    # up
            ])):
                neighbor_pos = pos + dir_vec
                neighbor_state = self.env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * self.env.width]
                
                # if neighbor state exists (not wall) 
                # and neighor state has been visited
                # and has higher eigenvector value
                # go to that neighbor state
                if neighbor_state >= 0 and self.visitation[neighbor_state] > 0  \
                        and self.eigvec[neighbor_state] > value:
                    value = self.eigvec[neighbor_state]
                    myopic_a = a

            if myopic_a == -1:
                # no better neighbor, terminate
                termination[s] = 1
            else:
                policy[s] = myopic_a

        self.myopic_policy = dict(termination=termination, policy=policy)
        return self.myopic_policy



    def plot_visitation_and_eigvec(self, e):
        """
        e: episode number
        Visualizes the current state visitation and eigenvec
        """
        # plot
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        plt.title("Visits")
        self.visualizer.visualize_shaping_reward_2d(self.visitation)

        plt.subplot(1, 3, 2)
        plt.title("Log Eigvec")
        self.visualizer.visualize_shaping_reward_2d(np.log(self.eigvec))

        # hill climbing eigenvector
        plt.subplot(1, 3, 3)
        policy = self.eigvec_myopic_policy()
        plt.title("Myopic Policy")
        self.visualizer.visualize_option_with_env_reward(policy)

        plt.suptitle(f"Episode {e}")


        plt.savefig(f"minigrid_basics/function_approximation/dr_eigvec_tabular_plots/{e}.png")
        plt.close()


if __name__ == "__main__":

    env_name = "two_goals"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()


    # create env
    seed = None
    np.random.seed(seed)
    env = gym.make(env_id, seed=seed, no_goal=False, no_start=False)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    # 
    learner = DR_Eigvec_Tabular(env, step_size=0.1, lambd=1.)
    learner.learn(n_episodes=5000)

    print(learner.eigvec)


    cs = np.array(learner.cosine_similarity).T

    plt.subplot(1, 2, 1)
    plt.plot(cs[0], label="true eigvec")
    plt.ylabel("Cosine Similarity")
    plt.subplot(1, 2, 2)
    plt.plot(cs[1], label="visitation")
    plt.xlabel("Number of Steps")
    
    # plt.ylim([-1, 1])
    plt.savefig("minigrid_basics/function_approximation/cs.png")

    # print(learner.eigvec)
    # print(learner.visitation)

    # save video
    os.chdir("minigrid_basics/function_approximation/dr_eigvec_tabular_plots")
    # for prefix in ['option', 'cumulative_visit', 'eigenvector']:
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'%d.png', '-r', '30','-pix_fmt', 'yuv420p', 
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-y', f'{env_name}_equal_tabular_eigvec_myopic_collection.mp4'
    ])

    for file_name in  glob.glob("*.png"):
        os.remove(file_name)



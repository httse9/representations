import numpy as np
import os
import gin
import gym
import matplotlib.pyplot as plt
from itertools import product
import types

# testing imports
import gin
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.q_learner import AuxiliaryReward
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class QLearner:
    """
    Run Q-Learning
    """
    def __init__(self, env, step_size, gamma=0.99, terminal_reward=[0, 0, 0, 0]):
        """
        env: environment
        aux_reward: AuxiliaryReward object to provide auxiliary rewards
        """
        self.env = env
        self.step_size = step_size
        self.gamma = gamma
        self.terminal_reward = terminal_reward

        # init Q values
        self.Q = np.zeros((env.num_states, env.num_actions))

    def epsilon_greedy_action_selection(self, state):
        """
        state: scalar
        epsilon: 0.05
        """
        if np.random.rand() < 0.1: # epsilon greedy
            a = np.random.choice(self.env.num_actions)
        else:
            a = np.argmax(self.Q[state])
        return a
    
    def update_Q(self, s, a, r, ns):
        """
        Update Q value given transition
        """
        self.Q[s, a] += self.step_size * (r + self.gamma * self.Q[ns].max() - self.Q[s, a])

    def get_current_policy(self):
        """
        Get policy from current Q values
        """
        pi = (self.Q == self.Q.max(1, keepdims=True)).astype(float)
        pi /= pi.sum(1, keepdims=True)
        return pi


    def learn(self, max_iter):
        """
        Learn.
        """

        env = self.env
        s = env.reset()

        for n in range(max_iter):
            # choose action
            # a = self.epsilon_greedy_action_selection(s['state'])
            a = np.random.choice(env.num_actions)

            # env step
            ns, r, done, d = env.step(a)
            terminated = d['terminated']

            if terminated:
                idx = self.env.terminal_idx.index(ns['state'])
                self.Q[ns['state']] = self.terminal_reward[idx]

            # update Q
            self.update_Q(s['state'], a, r, ns['state'])

            # reset environment if needed
            if done:
                s = env.reset()
            else:
                s = ns
       


class SuccessorFeatureLearner:

    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env

        self.gamma = gamma
        self.alpha = alpha

        self.sf = np.zeros((env.num_states, env.num_actions, 6))

        # init sf for goal states
        for i, ts in enumerate(self.env.terminal_idx):
            self.sf[ts, :, i + 2] = 1

    def learn(self, pi, n_steps=1000):

        s = self.env.reset()
        a = np.random.choice(self.env.num_actions)

        for i in range(n_steps):
            ns, r, done, d = self.env.step(a)

            # get state feature
            x, y = self.env.state_to_pos[s['state']]
            s_type = self.env.raw_grid[x, y]

            s_feat = [s_type in ['s', ' '], s_type == 'l', 0, 0, 0, 0]
            # for ts in self.env.terminal_idx:
            #     s_feat += [s['state'] == ts]
            s_feat = np.array(s_feat).astype(float)


            na_opt = np.random.choice(self.env.num_actions, p=pi[ns['state']])
            self.sf[s['state'], a] += self.alpha * (s_feat + self.gamma * self.sf[ns['state'], na_opt] - self.sf[s['state'], a])

            if done:
                s = self.env.reset()
            else:
                s = ns
            a = np.random.choice(self.env.num_actions)




class DefaultFeatureLearnerSA:
    """
    For r(s, a)
    """

    def __init__(self, env, alpha=0.1, lambd=1.3):
        self.env = env

        self.lambd = lambd
        self.alpha = alpha

        terminal_feat, weight = self.terminal_feat_and_weight()
        self.weight = weight
        self.df = np.zeros((env.num_states * env.num_actions, len(weight)))

        # init df
        for (idx, a), f in zip(product(self.env.terminal_idx, range(self.env.num_actions)), terminal_feat):
            # print(idx, a, f)
            # print(self.sa_to_idx(idx, a))
            self.df[self.sa_to_idx(idx, a)] = f

    def sa_to_idx(self, s, a):
        """
        Convert state-action pair (s,a) to index
        """
        return s * self.env.num_actions + a
        

    def terminal_feat_and_weight(self):
        weight = []
        for idx in self.env.terminal_idx:
            for a in range(self.env.num_actions):
                weight.append(np.exp(self.env.rewards[idx]))
        return np.eye(len(weight)), weight
    
    def learn(self, n_steps=1000):

        s = self.env.reset()
        a = np.random.choice(self.env.num_actions)

        for i in range(n_steps):

            ns, r, done, d = self.env.step(a)
            na = np.random.choice(self.env.num_actions)

            terminated = d['terminated']

            sa_idx = self.sa_to_idx(s['state'], a)      # idx of (s, a)
            nsa_idx = self.sa_to_idx(ns['state'], na)   # idx of (s', a')

            # learn df
            self.df[sa_idx] += self.alpha * (np.exp(r / self.lambd) * self.df[nsa_idx] - self.df[sa_idx])

            if done:
                s = self.env.reset()
                a = np.random.choice(self.env.num_actions)
            else:
                s = ns
                a = na

            
def new_terminal_weight_sa(rewards):
    weight = []
    for r in rewards:
        weight += [np.exp(r)] * 4

        # weight += [r] * 4
    return weight



if __name__ == "__main__":
    env_name = "fourrooms_multigoal"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(0)

    env = gym.make(env_id, seed=42, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    # s = env.reset()
    # print(s)
    # env.raw_grid[1, 4] = 's'
    # s = env.reset()
    # print(s)



    # env_eval.raw_grid[5, 9] = "s"
    # print(env_eval.raw_grid.T)

    #### learn optimal Q
    SFs = []        # set of SFs for different policies
    for terminal_reward in [[100, 0, 0, 0], [0, 0, 0, 100], [0, 100, 0, 0], [0, 0, 100, 0]]:
        qlearner = QLearner(env, 1.0, terminal_reward=terminal_reward)
        qlearner.learn(100000)
        pi =  qlearner.get_current_policy()

        sf_learner = SuccessorFeatureLearner(env)
        sf_learner.learn(pi, 100000)

        SFs.append(sf_learner.sf)



    # verify visually
    for i in range(4):
        sf_w = np.array([-1, -20, 0, 0, 0, 0])
        sf_w[i + 2] = 100

        # GPI
        Qs = [sf @ sf_w for sf in SFs]
        Qs = [q.reshape(env.num_states, 1, env.num_actions) for q in Qs]
        Qs = np.concatenate(Qs, axis=1)
        Qs = Qs.max(1)
        sf_pi = Qs.argmax(1)

        visualizer = Visualizer(env)
        policy = {}
        policy['policy'] = sf_pi
        policy['termination'] = np.zeros((env.num_states))
        for idx in env.terminal_idx:
            policy['termination'][idx] = 1.
        
        plt.subplot(1, 4, i + 1)
        visualizer.visualize_option(policy)
    plt.show()


    ###### DF for r(s, a)
    df_learner = DefaultFeatureLearnerSA(env)
    df_learner.learn(10000)


    new_df_rewards = np.eye(len(env.terminal_idx))  * 100

    for i, nr in enumerate(new_df_rewards):
        new_weight = new_terminal_weight_sa(nr)

        # Q = df_learner.df @ df_learner.weight
        Q = df_learner.df @ new_weight
        Q = Q.reshape(-1, 4)
        pi = Q.argmax(1)

        policy = {}
        policy['policy'] = pi
        policy['termination'] = np.zeros((Q.shape[0]))
        for idx in env.terminal_idx:
            policy['termination'][idx] = 1.

        plt.subplot(1, 4, i + 1)
        visualizer.visualize_option(policy)
    plt.show()


    # ###### DF for r(s)
    # df_learner = DefaultFeatureLearner(env,)

    # loss = df_learner.learn(1000000)

    # plt.plot(loss)
    # plt.show()

    # # print(df_learner.df)
    # optimal_v = df_learner.df @ df_learner.weight
    # print(optimal_v[df_learner.non_terminal_mask])

    # print(optimal_v[~df_learner.non_terminal_mask])

    # plt.subplot(1, 2, 1)
    # visualizer = Visualizer(env)
    # visualizer.visualize_shaping_reward_2d(np.log(optimal_v), ax=None, normalize=True, vmin=0, vmax=1)

    # plt.subplot(1, 2, 2)
    # optimal_v_true = df_learner.optimal_v_true
    # optimal_v[df_learner.non_terminal_mask] = optimal_v_true
    # visualizer.visualize_shaping_reward_2d(np.log(optimal_v), ax=None, normalize=True, vmin=0, vmax=1)
    # plt.show()

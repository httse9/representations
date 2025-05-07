import numpy as np
import os
import gin
import gym
import matplotlib.pyplot as plt
from itertools import product

# testing imports
import gin
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




def terminal_reward_features_and_weights(env):
    """
    Return 
    - terminal features
    - weights for reward function
    """
    num_terminal_states = len(env.terminal_idx)

    weight = []
    for i, idx in enumerate(env.terminal_idx):

        weight.append(np.exp(env.rewards[idx]))

    return np.eye(num_terminal_states), weight

class DefaultFeatureLearner:
    def __init__(self, env, alpha=0.1, lambd=1.3):
        self.env = env
        
        self.lambd = lambd
        self.alpha = alpha

        terminal_feat, weight = terminal_reward_features_and_weights(env)
        self.weight = weight
        self.terminal_feat = terminal_feat

        # init df
        self.df = np.zeros((env.num_states, len(weight)))
        for idx, feat in zip(env.terminal_idx, terminal_feat):
            self.df[idx] = feat


        # ground truth optimal v
        self.DR = self.compute_DR()

        non_terminal_mask = np.zeros((env.num_states))
        for idx in env.terminal_idx:
            non_terminal_mask[idx] = 1
        non_terminal_mask = (1 - non_terminal_mask) == 1
        self.non_terminal_mask = non_terminal_mask

        DR_NN = self.DR[non_terminal_mask][:, non_terminal_mask] 
        P_NT = self.P_pi[non_terminal_mask][:, ~non_terminal_mask]
        r_T = env.rewards[~non_terminal_mask]

        self.optimal_v_true = DR_NN @ P_NT @ np.exp(r_T)
        print(self.optimal_v_true)


    def compute_DR(self, pi = None):
        """
        Compute DR with respect to pi.
        If pi is None, use uniform random policy.
        """
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        R = self.env.rewards
        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)
        self.P_pi = P_pi
        DR = np.linalg.inv(np.diag(np.exp(-R / self.lambd)) - P_pi)

        return DR.copy()


    def learn(self, n_steps=1000):

        loss = []

        s = self.env.reset()

        for i in range(n_steps):
            a = np.random.choice(self.env.num_actions)

            ns, r, done, d = self.env.step(a)

            terminated = d['terminated']

            # if terminated:
            #     print("Goal Reached")

            # learn df
            assert s['state'] not in self.env.terminal_idx
            self.df[s['state']] += self.alpha * (np.exp(r / self.lambd) * self.df[ns['state']] - self.df[s['state']])

            if done:
                s = self.env.reset()
            else:
                s = ns

            if (i + 1) % 100 == 0:
                optimal_v_pred = self.df[self.non_terminal_mask] @ self.weight
                mse = ((optimal_v_pred - self.optimal_v_true) ** 2).mean()
                loss.append(mse)

        return loss
    
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

    ###### DF for r(s, a)
    df_learner = DefaultFeatureLearnerSA(env)
    df_learner.learn(200000)


    new_rewards = np.eye(len(env.terminal_idx))  * 100
    visualizer = Visualizer(env)

    

    for i, nr in enumerate(new_rewards):
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

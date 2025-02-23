import numpy as np
import gym
import matplotlib.pyplot as plt
import os

# testing imports
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from minigrid_basics.examples.rep_utils import *


class AuxiliaryReward:
    """
    """

    def __init__(self, env, r_aux, mode, r_aux_weight, gamma=0.99):
        """
        env: environment
        gamma: discount factor
        r_aux: auxiliary reward
        mode: potential, wang or none
        r_aux_weight: weight put on auxiliary reward

        Try not to modify r_aux...
        """
        self.env = env
        self.gamma = gamma
        self.r_aux = r_aux
        self.mode = mode
        assert 0 <= r_aux_weight <= 1
        self.r_aux_weight = r_aux_weight

        if mode == "wang":
            # Because of the square, magnitude of shaped reward can become quite large
            # Normalize such that auxiliary part has mean magnitude of 1.0
            wang_normalizer = np.square(r_aux - r_aux[env.terminal_idx[0]]).mean()
            self.r_aux = r_aux / np.sqrt(wang_normalizer)
            assert np.isclose(np.square(self.r_aux - self.r_aux[env.terminal_idx[0]]).mean(), 1.0)

    def shaped_reward(self, r_orig, state, next_state):
        """
        Return shaped reward based on the mode
        """
        if self.mode == "none":
            return r_orig
        elif self.mode == "wang":
            return self.wang_reward(r_orig, state, next_state)
        elif self.mode == "potential":
            return self.potential_shaped_reward(r_orig, state, next_state)

    def wang_reward(self, r_orig, state, next_state):
        """
        Return shaped reward described in Wang 2021 and Wu 2019's papers
        """
        # https://proceedings.mlr.press/v139/wang21ae/wang21ae.pdf (Section 4.3)
        # https://openreview.net/pdf?id=HJlNpoA5YQ (Section 5.2)

        # get idx of terminal state
        terminal_idx = self.env.terminal_idx[0]

        return (1 - self.r_aux_weight) * r_orig - self.r_aux_weight * np.square(self.r_aux[terminal_idx] - self.r_aux[next_state])

    def potential_shaped_reward(self, r_orig, state, next_state):
        """
        Potential-based reward shaping

        Params:
        r_orig: reward of the original MDP
        state: scalar, the current state
        next state: scalar, the next state
        """
        # average of r_orig and potential reward of r_aux
        return (1 - self.r_aux_weight) * r_orig + self.r_aux_weight * (self.gamma * self.r_aux[next_state] - self.r_aux[state])


class QLearner:
    """
    Run Q-Learning
    """
    def __init__(self, env, env_eval, aux_reward: AuxiliaryReward, step_size, gamma=0.99):
        """
        env: environment
        aux_reward: AuxiliaryReward object to provide auxiliary rewards
        """
        self.env = env
        self.env_eval = env_eval
        self.aux_reward = aux_reward
        self.step_size = step_size
        self.gamma = gamma

        # init Q values
        self.Q = np.zeros((env.num_states, env.num_actions))

    def epsilon_greedy_action_selection(self, state):
        """
        state: scalar
        epsilon: 0.05
        """
        if np.random.rand() < 0.05: # epsilon greedy
            a = np.random.choice(self.env.num_actions)
        else:
            a = np.argmax(self.Q[state])
        return a
    
    def update_Q(self, s, a, r, ns, terminated):
        """
        Update Q value given transition
        """
        if terminated:
            self.Q[s, a] += self.step_size * (r - self.Q[s, a])
        else:
            self.Q[s, a] += self.step_size * (r + self.gamma * self.Q[ns].max() - self.Q[s, a])

    def get_current_policy(self):
        """
        Get policy from current Q values
        """
        pi = (self.Q == self.Q.max(1, keepdims=True)).astype(float)
        pi /= pi.sum(1, keepdims=True)
        return pi

    def eval_policy(self):
        """
        Evaluate current policy and return undiscounted return.
        Env is deterministic, policy also (not exactly but shouldn't matter much). One episode suffices.
        """
        pi = self.get_current_policy()

        env = self.env_eval
        s = env.reset()
        ret = 0     # return

        done = False
        while not done:
            a = np.random.choice(env.num_actions, p=pi[s['state']])
            s, r, done, _ = env.step(a)
            ret += r

        return ret

    def learn(self, max_iter, log_interval):
        """
        Learn.
        """

        env = self.env
        s = env.reset()

        timesteps = [0]
        returns = [self.eval_policy()]
        Qs = [self.Q.copy()]

        for n in range(max_iter):
            # choose action
            a = self.epsilon_greedy_action_selection(s['state'])

            # env step
            ns, r, done, d = env.step(a)
            terminated = d['terminated']

            # compute shaped reward
            shaped_reward = self.aux_reward.shaped_reward(r, s['state'], ns['state'])

            # update Q
            self.update_Q(s['state'], a, shaped_reward, ns['state'], terminated)

            # reset environment if needed
            if done:
                s = env.reset()
            else:
                s = ns

            # eval policy regularly
            if (n + 1) % log_interval == 0:
                timesteps.append(n + 1)
                returns.append(self.eval_policy())

                if (n + 1) % 1000 == 0:
                    Qs.append(self.Q.copy())

        return np.array(timesteps), np.array(returns), np.array(Qs)

            
### test
if __name__ == "__main__":

    env_name = "gridmaze_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(42)

    env = gym.make(env_id, seed=42)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    env_eval = gym.make(env_id, seed=42)
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)

    ### test no reward shaping (good, it learns)
    aux_reward = AuxiliaryReward(env, None, "none")
    qlearner = QLearner(env, env_eval, aux_reward, 0.1)

    # t, ret = qlearner.learn(10000, 100)
    # plt.plot(t, ret, label="no shaping")
    # plt.ylim([-200, None])
    # plt.show()


    ### test reward shaping wang
    shaper = RewardShaper(env)
    eigvec_SR = shaper.SR_top_eigenvector()
    reward_SR = shaper.shaping_reward_transform_using_terminal_state(eigvec_SR)
    aux_reward = AuxiliaryReward(env, reward_SR, "wang")
    qlearner = QLearner(env, env_eval, aux_reward, 1.0)

    plot_value_pred_map(env, aux_reward.r_aux, contain_goal_value=True)
    plt.show()

    # t, ret = qlearner.learn(100000, 10000)
    # plt.plot(t, ret, label="SR wang")
    # plt.ylim([-200, None])
    # plt.show()

    # ### test reward shaping SR potential
    aux_reward = AuxiliaryReward(env, reward_SR, "potential")
    qlearner = QLearner(env, env_eval, aux_reward, 1.0)

    plot_value_pred_map(env, aux_reward.r_aux, contain_goal_value=True)
    plt.show()

    # t, ret = qlearner.learn(100000, 10000)
    # plt.plot(t, ret, label="SR potential")
    # # plt.ylim([-200, None])
    # plt.show()

    # ### test reward shaping DR potential
    eigenvec_DR = shaper.DR_top_log_eigenvector(lambd=1.3)
    reward_DR = shaper.shaping_reward_transform_using_terminal_state(eigenvec_DR)
    aux_reward = AuxiliaryReward(env, reward_DR, "potential")
    qlearner = QLearner(env, env_eval, aux_reward, 0.1)

    plot_value_pred_map(env, aux_reward.r_aux, contain_goal_value=True)
    plt.show()

    # t, ret = qlearner.learn(10000, 100)
    # plt.plot(t, ret, label = "DR potential")

    # plt.legend()
    # # plt.xscale('log')
    # # plt.ylim([-1000, None])
    # plt.show()

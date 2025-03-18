import numpy as np
from minigrid_basics.examples.rep_utils import *
from tqdm import tqdm
import matplotlib.pylab as plt
import os
import random
from os.path import join
import pickle

import warnings
warnings.filterwarnings("error")

class TD_Learner:
    """
    Generic class for learning the value funcion using TD.
    """
    def __init__(self, env, env_eval, features, tabular=False, learn_features=False, gamma=0.99, epsilon=0.05, v_lr=0.0001, \
                 lambd=1, seed=0, log_process=False):
        """
        Params:
        - env: environment
        - features: what features to use
            - type: string
            - ["SR", "DR", "MER"]

        - tabular: whether 
        - learn_features: whether to also learn features while learning V
        - gamma: only used for the SR
        - epsilon: epsilon greedy, only used for the SR (DR and MER use stochastic policies)

        - v_lr: learning rate for learning the value function
        """
        self.env = env
        self.env_eval = env_eval
        self.gamma = gamma
        self.features = features
        assert self.features in ["SR", "DR", "MER"]
        self.tabular = tabular
        self.learn_features = learn_features
        assert not (self.learn_features and self.tabular)

        self.epsilon = epsilon
        self.v_lr = v_lr
        self.log_process = log_process

        self.create_save_path()

        # lambda that controls the relative weight of the KL or entropy term
        if self.features == "SR":
            # lambda only used for DR and MER
            assert lambd == 1
        self.lambd = lambd
        self.env.rewards /= self.lambd      # scale all rewards by 1 / lambda

        # initialize features and weights
        self.set_seed(seed)
        self.init_features()
        self.init_weights()
        # assert (self.F >= 0).all()
        

    def create_save_path(self,):
        if self.tabular:
            name = self.features + "_tabular"
        else:
            name = self.features
        self.save_path = join('minigrid_basics', 'examples', 'rep_plots', 'td_learn', self.env.unwrapped.spec.id, name)
        os.makedirs(self.save_path, exist_ok=True)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        print("Current seed:", self.seed)

    def init_features(self):
        if self.tabular:
            # identity feature equiv. to tabular
            self.F = np.eye(self.env.num_states)
        elif self.learn_features:
            # if learn features, init to all 0s
            self.F = np.zeros((self.env.num_states, self.env.num_states))
        else:
            # use SR, DR or MER as features
            self.F = get_representation(self.env, self.features)
            self.process_features()

    def process_features(self):
        pass
        # for DR, MER, mathematically, we only need entries for non-terminal states
        # self.F = self.F[:, self.env.nonterminal_idx]


        # if self.log_process:
        #     F_nonterminal = self.F[self.env.nonterminal_idx]
        #     F_nonterminal = np.log(F_nonterminal)

        #     self.F[self.env.nonterminal_idx] = F_nonterminal

        # col_max = np.abs(self.F[self.env.nonterminal_idx]).max(axis=0, keepdims=True)
        # self.F /= col_max

        # print(np.abs(self.F[self.env.nonterminal_idx]).max(0))

    def update_features(self):
        """
        Update features using a transition.
        """
        pass

    def update_SR(self):
        pass


    def update_DR(self):
        pass

    def update_MER(self):
        pass

    def init_weights(self, mode="ones"):
        """
        Initialize weights for value approximation.
        """
        n_features = self.F.shape[1]
        if mode == "rand":
            # self.w = np.random.normal(size=(n_features))
            self.w = np.random.rand(n_features)
        elif mode == "zeros":
            self.w = np.zeros((n_features))
        elif mode == 'ones':
            self.w = np.ones((n_features))

    def update_weights(self, s, action_prob, r, ns, terminated):
        """
        Update weights using a transition
        """
        if self.features == "SR":
            self.update_weights_SR(s, r, ns, terminated)
        elif self.features == "DR":
            self.update_weights_DR(s, action_prob, r, ns, terminated)
        elif self.features == "MER":
            self.update_weights_MER(s, action_prob, r, ns, terminated)

    def update_weights_SR(self, s, r, ns, terminated):
        """
        
        """
        # value of curr state
        v_s = self.F[s] @ self.w

        # value of next state
        if terminated:
            v_ns = 0
        else:
            v_ns = self.F[ns] @ self.w

        td_error = r + self.gamma * v_ns - v_s
        self.w += self.v_lr * td_error * self.F[s]

    def update_weights_DR(self, s, action_prob, r, ns, terminated):
        """
        
        """
        # z of curr state
        z_s = self.F[s] @ self.w

        # z of next state
        if terminated:
            z_ns = 1
        else:
            z_ns = self.F[ns] @ self.w

        is_ratio = 1 / self.env.num_actions / action_prob   # importance sampling ratio
        td_error = is_ratio * np.exp(r / self.lambd) * z_ns - z_s
        # print(self.v_lr * td_error * self.F[s])
        # print(np.abs(self.v_lr * td_error * self.F[s]).max())
        # print(td_error)

        self.w += self.v_lr * td_error * self.F[s]
        # mathematically, w should be all non-negative, we explicitly constrain entries of w to be non-negative
        if (self.w < 0).any():
            # self.w -= self.w.min()
            # this works much better than the one above
            self.w[self.w < 0] = 0

        # print(self.F[s])
        # print(self.w)

    def update_weights_MER(self, s, action_prob, r, ns, terminated):
        """
        
        """
        # z of curr state
        z_s = self.F[s] @ self.w

        # z of next state
        if terminated:
            z_ns = 1
        else:
            z_ns = self.F[ns] @ self.w

        is_ratio = 1 / self.env.num_actions / action_prob   # importance sampling ratio
        td_error = is_ratio * self.env.num_actions * np.exp(r / self.lambd) * z_ns - z_s

        self.w += self.v_lr * td_error * self.F[s]

        # mathematically, w should be all non-negative
        # we explicitly constrain entries of w to be non-negative
        if (self.w < 0).any():
            # self.w -= self.w.min()

            # this works much better than the one above
            self.w[self.w < 0] = 0


    def policy_improvement(self):
        """
        Update policy using current approximated value function≥
        """
        if self.features == "SR":
            return self.policy_improvement_SR()
        elif self.features == "DR":
            return self.policy_improvement_DR()
        elif self.features == "MER":
            return self.policy_improvement_MER()

    def policy_improvement_SR(self):
        """
        
        """
        V = self.F @ self.w
        P = self.env.transition_probs
        Q = P @ V
        pi_new = (Q == Q.max(1, keepdims=True)).astype(float)
        pi_new /= pi_new.sum(1, keepdims=True)  
        return pi_new

    def policy_improvement_DR(self):
        """
        Assumes uniform random default policy
        """
        Z = self.F @ self.w
        P = self.env.transition_probs
        Q = P @ Z
        Q[self.env.nonterminal_idx] /= Q[self.env.nonterminal_idx].sum(1, keepdims=True)

        return Q

    def policy_improvement_MER(self):
        Z = self.F @ self.w
        P = self.env.transition_probs
        Q = P @ Z
        Q /= Q.sum(1, keepdims=True)
        return Q


    def compute_optimal_V(self):
        """
        Compute optimal value function.
        Optimal value function differs according to the representation used.
        """
        if self.features == "SR":
            return self.compute_optimal_V_SR()
        elif self.features == "DR":
            return self.compute_optimal_V_DR()
        elif self.features == "MER":
            return self.compute_optimal_V_MER()

    def compute_optimal_V_SR(self):
        """
        VI to get ground truth value function
        P: transition prob matrix (S, A, S)
        R: reward vector, (S)
        """
        # cannot do closed form using SR,
        # because need SR wrt to optimal policy,
        # not SR wrt to uniform random policy.
        return value_iteration(self.env, self.gamma)

    def compute_optimal_V_DR(self):
        """
        Compute the optimal value function in linearly solvable MDPS.
        Assumes that the default policy is uniform random.
        """
        R = self.env.rewards
        P = self.env.transition_probs
        nonterminal_idx = (P.sum(-1).sum(-1) != 0)

        # 1st Method: compute optimal V in closed form using DR
        DR = get_representation(self.env, "DR")
        P_NT = P.mean(1)[nonterminal_idx][:, ~nonterminal_idx]
        z_closed = np.ones((self.env.num_states))   # init as 1, since terminal states z value = 1
        z_closed[nonterminal_idx] = DR[nonterminal_idx][:, nonterminal_idx] @ P_NT @ np.exp(R)[~nonterminal_idx]

        # print(P_NT @ np.exp(R)[~nonterminal_idx])

        return np.log(z_closed) * self.lambd

        # 2nd Method: Compute optimal V using z iteration.
        # Since the values are very near 0, checking diff between old and new z as termination conditino does not work well.
        z = np.ones((self.env.num_states))
        for i in range(2000):   # need to ensure this large enough
            z[nonterminal_idx] = np.exp(R)[nonterminal_idx].reshape(-1, 1) * P.mean(1)[nonterminal_idx] @ z

        assert np.allclose(z_closed, z)
        return np.log(z)

    def compute_optimal_V_MER(self):
        """
        Compute the optimal value function for maximum entropy RL.
        """
        R = self.env.rewards
        P = self.env.transition_probs
        nonterminal_idx = (P.sum(-1).sum(-1) != 0)

        # 1st Method: compute in closed form using MER
        MER = get_representation(self.env, "MER")
        P_NT = P.mean(1)[nonterminal_idx][:, ~nonterminal_idx]
        z_closed = np.ones((self.env.num_states))   # init as 1, since terminal states z value = 1
        z_closed[nonterminal_idx] = MER[nonterminal_idx][:, nonterminal_idx] @ P_NT @ np.exp(R)[~nonterminal_idx]

        return np.log(z_closed) * self.lambd

        # 2nd Method: compute using z iteration
        z = np.ones((self.env.num_states))
        for i in range(2000):
            z[nonterminal_idx] = np.exp(R)[nonterminal_idx].reshape(-1, 1) * P.sum(1)[nonterminal_idx] @ z
        assert np.allclose(z_closed, z)
        return np.log(z)

    def eval_policy(self, pi, n_episodes=10):
        """
        Evaluate performance of policy.

        Do not do discount!
        """
        return_list = []
        for n in range(n_episodes):

            s = self.env_eval.reset()

            done = False
            discount = 1
            episode_return = 0

            while not done:
                a = np.random.choice(self.env_eval.num_actions, p=pi[s['state']])

                s, r, done, _ = self.env_eval.step(a)
                # print(s)

                episode_return += discount * r
                # discount *= self.gamma

            return_list.append(episode_return)

        print(return_list)
        return np.mean(return_list)

    def learn_features_TD(self):
        """
        Learn SR, DR, or MER by TD
        """
        pass

    def select_action(self, s):
        if self.features == "SR":
            return self.select_action_SR(s)
        elif self.features == "DR":
            return self.select_action_DR(s)
        elif self.features == "MER":
            return self.select_action_MER(s)

    def select_action_SR(self, s):
        """
        Returns selected action, and the corresponding action probability.
        Since for the SR, the action probability is not used, return None.
        """
        V = self.F @ self.w
        p = self.env.transition_probs[s]
        q = p @ V
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.num_actions), None
        else:
            return np.argmax(q), None

    def select_action_DR(self, s):
        """
        Return selected action, and corresponding prob.
        """
        Z = self.F @ self.w
        p = self.env.transition_probs[s]
        q = p @ Z
        try:
            q /= q.sum()
        except:
            print(self.w)

        a = np.random.choice(self.env.num_actions, p=q)
        prob = q[a]
        return a, prob

    def select_action_MER(self, s):
        """
        Return selected action, and corresponding prob.
        """
        Z = self.F @ self.w
        p = self.env.transition_probs[s]
        q = p @ Z
        q /= q.sum()

        a = np.random.choice(self.env.num_actions, p=q)
        prob = q[a]
        return a, prob

    def learn(self, max_iter, log_interval = 10000, seed=0):
        """
        Learn the optimal value function using TD.
        Collect transitions using policy induced by current approximated V.

        Params:
        - max_iter: maximum number of iterations to run
        """

        self.set_seed(seed)
        self.init_features()
        self.init_weights()
        # assert (self.F >= 0).all()

        s = self.env.reset()

        data = {
            'n_iter': [],
            'return': [],
            'mse': []
        }

        V_optimal = self.compute_optimal_V()
        def mse(V_optimal, V_curr):
            return ((V_optimal[self.env.nonterminal_idx] - V_curr[self.env.nonterminal_idx]) ** 2).mean()

        pbar = tqdm(total=max_iter)
        for n in range(max_iter):
            
            # select action
            a, action_prob = self.select_action(s['state'])

            # execute action
            ns, r, done, d = self.env.step(a)

            # update weights
            self.update_weights(s['state'], action_prob, r, ns['state'], d['terminated'])

            if done:
                s = self.env.reset()
            else:
                s = ns

            if (n + 1) % log_interval == 0:
                pi = self.policy_improvement()

                ret = self.eval_policy(pi, n_episodes=10)
                pbar.set_description(f"Return: {ret}")
                pbar.update(log_interval)

                if self.features == "SR":
                    curr_V = self.F @ self.w
                else:
                    curr_Z = self.F @ self.w
                    curr_Z[self.env.nonterminal_idx] = np.log(curr_Z[self.env.nonterminal_idx]) * self.lambd
                    curr_V = curr_Z

                data['n_iter'].append(n + 1)
                data['return'].append(ret)
                data['mse'].append(mse(V_optimal, curr_V))

                print(data['mse'][-1])
                # print(self.w)

                # plot_value_pred_map(self.env, curr_V, True)
                # plt.show()



        self.save_td_learn_data(data)

        return data
    
    def save_td_learn_data(self, data):
        """
        File structure:
        {
            'seed': [0, 1, 2],
            'data': [{data0}, {data1}, {data2}]
        }
        """

        file_name = f"{self.v_lr}.pkl"

        # check if exists
        try:
            with open(join(self.save_path, file_name), "rb") as f:
                data_dict = pickle.load(f)

            # if exists, check seed
            if self.seed in data_dict['seed']:
                print(f"Data for seed {self.seed} already exists")

                # replace data
                print("Replace old data with new.")
                idx = data_dict['seed'].index(self.seed)
                data_dict['data'][idx] = data
    
            else:
                # data for self.seed do not exist, add to data
                data_dict['seed'].append(self.seed)
                data_dict['data'].append(data)

        except:
            print("Data dict does not exist. Creating...")
            data_dict = {}
            data_dict['seed'] = [self.seed]
            data_dict['data'] = [data]

        with open(join(self.save_path, file_name), "wb") as f:
            pickle.dump(data_dict, f)

        print(f"Data for seed {self.seed} saved.")        

        







    

def construct_value_pred_map(env, value_prediction, contain_goal_value=False):
    """
    Take the vector of predicted values, and visualize in the environment map.

    Params:
    value_prediction: vector of predicted state values, shape (S)
    contain_goal_value: whether the input value_prediction contains the value \ 
        prediction of goal values. If False, 
    """
    state_num = 0
    value_pred_map = np.zeros(env.reward_grid.shape) - float('-inf')

    for i in range(env.height):
        for j in range(env.width):
            if env.reward_grid.T[i, j] == 1:
                # skip wall
                continue

            if not contain_goal_value:
                if env.raw_grid.T[i, j] == 'g':
                    value_pred_map[i, j] = env.reward_grid.T[i, j]
                    continue


            value_pred_map[i, j] = value_prediction[state_num]
            state_num += 1

    return value_pred_map

def plot_value_pred_map(env, value_prediction, contain_goal_value=False, v_range=(None, None)):
    map = construct_value_pred_map(env, value_prediction, contain_goal_value=contain_goal_value)

    vmin, vmax = v_range
    plt.imshow(map, vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])

    # matplotlib.rcParams.update({'font.size': 5})
    for (j, i), label in np.ndenumerate(map):
        if not np.isinf(label):
            plt.text(i, j, np.round(label, 1), ha='center', va='center', color='white', \
                     fontsize= 'xx-small')
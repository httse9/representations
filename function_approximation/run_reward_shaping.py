import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import gin
import argparse
import random
import pickle
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.examples.q_learner import AuxiliaryReward, QLearner

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def create_aux_reward(env, env_name, mode, r_aux_weight):
    # create reward shaper
    shaper = RewardShaper(env)

    if mode == "none":
        # no reward shaping
        aux_reward = AuxiliaryReward(env, None, "none", 0)

    elif mode == "SR":
        # SR for potential based reward shaping
        assert 0 < r_aux_weight <= 1
        eigvec_SR = shaper.SR_top_eigenvector()
        reward_SR = shaper.normalize(eigvec_SR)
        aux_reward = AuxiliaryReward(env, reward_SR, "potential", r_aux_weight)

    elif mode == "DR":
        # DR for potential based reward shaping
        assert 0 < r_aux_weight <= 1
        eigvec_DR = shaper.DR_top_log_eigenvector(lambd=1.3)
        reward_DR = shaper.normalize(eigvec_DR)
        aux_reward = AuxiliaryReward(env, reward_DR, "potential", r_aux_weight)

    elif mode == "WGL":
        # reward-weighted graph laplacian
        assert 0 < r_aux_weight <= 1
        eigvec_WGL = shaper.WGL_smallest_eigenvector(lambd=20)
        reward_WGL = shaper.normalize(eigvec_WGL)
        aux_reward = AuxiliaryReward(env, reward_WGL, "potential", r_aux_weight)
        

    else:
        raise ValueError(f"Mode '{mode}' not recognized.")
    
    return aux_reward
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms", help="Environment")

    # reward shaping related
    parser.add_argument("--mode", default="none", help="Mode of reward shaping. CHoose from [none, SR, DR]")
    parser.add_argument("--r_aux_weight", default=0., type=float, help="Weight for convex combination of original reward and auxiliary reward.")

    # Q Learning related
    parser.add_argument("--step_size", default=0.1, type=float, help="Step size")
    parser.add_argument("--max_iter", default=100000, type=int, help="Number of steps to run Q-learning")
    parser.add_argument("--log_interval", default=100, type=int, help="Evaluate current policy every [log_interval] steps.")

    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    args = parser.parse_args()

    # set random seed
    set_random_seed(args.seed)

    # create environment
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    # environment for training
    env = gym.make(env_id, seed=args.seed)

    ### NOTE: Handle env creation differently
    # In the DR paper, terminal states are formulated as states that transition to an absorbing state,
    # and P(s' | s) = 0 for all s' for a terminal s.
    # In the WGL paper, terminal states are formulated as absorbing states,
    # so P(s | s) = 1 for terminal s.
    # To compare with the DR paper, we follow their formulation for SR and DR.
    if args.mode == "WGL":
        env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)
    else:
        env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=False)

    # separate environment for evaluation
    env_eval = gym.make(env_id, )
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)

    # create auxiliary reward
    aux_reward = create_aux_reward(env, args.env, args.mode, args.r_aux_weight)

    # create QLearner
    qlearner = QLearner(env, env_eval, aux_reward, args.step_size)

    # learn
    t, ret, Qs = qlearner.learn(args.max_iter, args.log_interval)
    # plt.plot(t, ret, label=args.mode)
    # plt.show()

    # save result
    path = os.path.join("minigrid_basics", "experiments", "reward_shaping", args.env, args.mode, )
    os.makedirs(path, exist_ok=True)
    filename = f"{args.r_aux_weight}-{args.step_size}-{args.seed}.pkl"

    data = dict(
        t = t,
        ret = ret,
        Qs=Qs
    )

    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(data, f)

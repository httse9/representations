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

def get_lambd(env_name):
    return 1.3
    # envs = [
    #     'dayan', 'dayan_2',
    #     'fourrooms', 'fourrooms_2',
    #     'gridroom', 'gridroom_2',
    #     'gridmaze', 'gridmaze_2'
    # ]
    # lambds = [
    #     1, 1, 1, 1, 1, 1.1, 1.1, 1.3
    # ]

    # idx = envs.index(env_name)
    # return lambds[idx]

def create_aux_reward(env, env_name, mode):
    # create reward shaper
    shaper = RewardShaper(env)

    if mode == "none":
        # no reward shaping
        aux_reward = AuxiliaryReward(env, None, "none")

    elif mode == "SR_wang":
        # SR reward shaping in prev work
        eigvec_SR = shaper.SR_top_eigenvector()
        reward_SR = shaper.shaping_reward_transform_using_terminal_state(eigvec_SR)
        aux_reward = AuxiliaryReward(env, reward_SR, "wang")

    elif mode == "SR_potential":
        # SR for potential based reward shaping
        eigvec_SR = shaper.SR_top_eigenvector()
        reward_SR = shaper.shaping_reward_transform_using_terminal_state(eigvec_SR)
        aux_reward = AuxiliaryReward(env, reward_SR, "potential")

    elif mode == "DR_potential":
        # DR for potential based reward shaping
        eigvec_DR = shaper.DR_top_log_eigenvector(lambd=get_lambd(env_name))
        reward_DR = shaper.shaping_reward_transform_using_terminal_state(eigvec_DR)
        aux_reward = AuxiliaryReward(env, reward_DR, "potential")

    else:
        raise ValueError(f"Mode '{mode}' not recognized.")
    
    return aux_reward
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms", help="Environment")

    # reward shaping related
    parser.add_argument("--mode", default="none", help="Mode of reward shaping. CHoose from [none, SR_wang, SR_potential, DR_potential]")

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
    env = maxent_mdp_wrapper.MDPWrapper(env)

    # separate environment for evaluation
    env_eval = gym.make(env_id, )
    env_eval = maxent_mdp_wrapper.MDPWrapper(env_eval)

    # create auxiliary reward
    aux_reward = create_aux_reward(env, args.env, args.mode)

    # create QLearner
    qlearner = QLearner(env, env_eval, aux_reward, args.step_size)

    # learn
    t, ret, Qs = qlearner.learn(args.max_iter, args.log_interval)
    # plt.plot(t, ret, label=args.mode)
    # plt.show()

    # save result
    path = os.path.join("minigrid_basics", "experiments", "reward_shaping", args.env, args.mode, )
    os.makedirs(path, exist_ok=True)
    filename = f"{args.step_size}-{args.seed}.pkl"

    data = dict(
        t = t,
        ret = ret,
        Qs=Qs
    )

    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(data, f)

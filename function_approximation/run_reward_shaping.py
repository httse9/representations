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
from minigrid_basics.function_approximation.reward_shaper import RewardShaper
from minigrid_basics.function_approximation.q_learner import AuxiliaryReward, QLearner
from os.path import join
import wandb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def create_aux_reward(env, env_name, mode, r_aux_weight, seed):
    # create reward shaper
    shaper = RewardShaper(env)

    if mode == "none":
        # no reward shaping
        aux_reward = AuxiliaryReward(env, None, "none", r_aux_weight)

    elif mode == "SR":
        # SR for potential based reward shaping
        assert 0 < r_aux_weight <= 1
        eigvec_SR = shaper.SR_top_eigenvector()
        reward_SR = shaper.normalize(eigvec_SR)
        aux_reward = AuxiliaryReward(env, reward_SR, "potential", r_aux_weight)

    elif mode == "DR":
        # DR for potential based reward shaping
        assert 0 < r_aux_weight <= 1

        # path of eigenvector learned using neural net with image inputs
        path = join("minigrid_basics", "function_approximation", "experiments_dr_anchor_real", env_name, "image", "data")
        with open(join(path, f"20.0-1e-05-1e-05-2000-0-rmsprop-dr_anchor-0.5-{seed}.pkl"), "rb") as f:
            data = pickle.load(f)

        eigvec_DR = data['eigvec']
        eigvec_DR -= np.log(np.linalg.norm(np.exp(eigvec_DR)))  # normalize
        reward_DR = shaper.normalize(eigvec_DR) # further normalize
        aux_reward = AuxiliaryReward(env, reward_DR, "potential", r_aux_weight)

    else:
        raise ValueError(f"Mode '{mode}' not recognized.")
    
    return aux_reward
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms", help="Environment")

    # reward shaping related
    parser.add_argument("--mode", default="none", help="Mode of reward shaping. CHoose from [none, SR, DR]")
    parser.add_argument("--r_aux_weight", default=0.5, type=float, help="Weight for convex combination of original reward and auxiliary reward.")

    # Q Learning related
    parser.add_argument("--step_size", default=0.1, type=float, help="Step size")
    parser.add_argument("--max_iter", default=100000, type=int, help="Number of steps to run Q-learning")
    parser.add_argument("--log_interval", default=100, type=int, help="Evaluate current policy every [log_interval] steps.")

    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    args = parser.parse_args()

    if args.mode == "none":
        # if no reward shaping, do not weight aux cause no aux reward
        args.r_aux_weight = 0

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
    aux_reward = create_aux_reward(env, args.env, args.mode, args.r_aux_weight, args.seed)

    # create QLearner
    qlearner = QLearner(env, env_eval, aux_reward, args.step_size)

    # learn
    t, ret, Qs, vlr = qlearner.learn(args.max_iter, args.log_interval)
    # plt.plot(t, ret, label=args.mode)
    # plt.show()

    group_name = f"{args.env}-{args.mode}"
    run_name = f"{args.r_aux_weight}-{args.step_size}-{args.seed}"

    run = wandb.init(
        project=f"reward-shaping",
        config=vars(args),
        group=group_name,  
        job_type="train",
        name=run_name, 
    )
    run.define_metric("train/*", step_metric="train/step")

    for tt, rett in zip(t, ret):
        wandb.log({
            "train/step": tt,
            "train/return": rett
        })

    wandb.log({"visit_low_reward": vlr})

    run.finish()

    # # save result
    # path = os.path.join("minigrid_basics", "function_approximation", "experiments_rs", args.env, args.mode, )
    # os.makedirs(path, exist_ok=True)
    # filename = f"{args.step_size}-{args.seed}.pkl"

    # data = dict(
    #     t = t,
    #     ret = ret,
    #     Qs=Qs
    # )

    # with open(os.path.join(path, filename), "wb") as f:
    #     pickle.dump(data, f)

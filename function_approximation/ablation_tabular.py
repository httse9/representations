import numpy as np
import random
import pickle
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.examples.reward_shaper import RewardShaper
import matplotlib.pyplot as plt
import os
import subprocess
import glob
from minigrid_basics.function_approximation.eigenlearner import *
from os.path import join
from tqdm import tqdm
from copy import deepcopy
import wandb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_batches(batch_size, key, *datasets):

    N = datasets[0].shape[0]
    idx = random.permutation(key, N)

    for start in range(0, N, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield tuple(ds[batch_idx] for ds in datasets)


def load_static_dataset(args):
    if args.dr_mode == "dr_anchor":
        with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_state_num_2.pkl", "rb") as f:
            dataset = pickle.load(f)
    elif args.dr_mode in ["dr_norm", "dr_gdo", "dr_gdo_log"]:
        with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_state_num.pkl", "rb") as f:
            dataset = pickle.load(f)

    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_state_num_test.pkl", "rb") as f:
        test_set = np.array(pickle.load(f))

    return dataset, test_set
    

def compute_l_GDO_grad(v, s, r, ns, lambd, barrier):
    return np.exp(-r / lambd) * v[s] - v[ns] + barrier * ((v ** 2).sum() - 1) * v[s]

def compute_tilde_l_GDO_grad(v, s, r, ns, lambd, barrier):
    u = np.exp(v)
    return np.exp(-r / lambd) * u[s] - u[ns] + barrier * ((u ** 2).sum() - 1) * u[s]

def compute_hat_l_GDO_grad(v, s, r, ns, lambd, barrier):
    return np.exp(-r / lambd) - np.exp(v[ns] - v[s]) + barrier * (np.exp(2 * v).sum() - 1)

def compute_l_DROGO_grad(v, s, r, ns, t, lambd):
    return np.exp(-r / lambd) - np.exp(v[ns] * (1 - t) - v[s])

def compute_cos_sim(v1, v2):
    return v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


def eigenlearning_tabular(args, env, ):
    
    dataset, _ = load_static_dataset(args)
    n = len(dataset)

    shaper = RewardShaper(env)
    eigvec_gt = shaper.DR_top_log_eigenvector(lambd=args.lambd) # ground-truth
    eigvec_gt -= eigvec_gt.max()


    # obs, actions, rewards, next_obs, next_rewards, terminals = [np.array(x) for x in zip(*dataset)]
    # rewards /= args.lambd
    # next_rewards /= args.lambd

    # v = np.ones(env.num_states)
    v = np.random.normal(size=(env.num_states))
    if args.dr_mode == "ar_anchor":
        v[env.terminal_idx[0]] = 0

    cos_sims = [compute_cos_sim(eigvec_gt, v)]
    es = [0]

    for e in tqdm(range(50000)):

        for (s, a, r, ns, nr, t) in dataset:
            
            if args.dr_mode == "dr_gdo":
                ### GDO 
                grad_s = compute_l_GDO_grad(v, s, r, ns, args.lambd, args.barrier)

            elif args.dr_mode == "dr_gdo_log":
                ### GDO Log Parameterization
                grad_s = compute_tilde_l_GDO_grad(v, s, r, ns, args.lambd, args.barrier)

            elif args.dr_mode == "dr_norm":
                ### GDO Log Param. Natural Gradient
                grad_s = compute_hat_l_GDO_grad(v, s, r, ns, args.lambd, args.barrier)

            elif args.dr_mode == "dr_anchor":
                ### GDO Log Param. Nat. Grad. Anchor
                grad_s = compute_l_DROGO_grad(v, s, r, ns, t, args.lambd,)

            v[s] -= 0.003 * grad_s

        if not (e + 1) % 100:
            es.append(e + 1)
            if args.dr_mode == "dr_gdo":
                # compute cos sim between ground-truth and logged u
                cos_sims.append(compute_cos_sim(eigvec_gt, np.log(np.abs(v))))
            
            else:
                cos_sims.append(compute_cos_sim(eigvec_gt, v))      


    for e, cs in zip(es, cos_sims):
        wandb.log({
            "train/epoch": e,
            "train/cosine_simlarity": cs
        })

    # if args.dr_mode == "dr_gdo":
    #     if v.sum() < 0:
    #         v *= -1
    #     visualizer.visualize_shaping_reward_2d(np.log(v), ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    # else:
    #     visualizer.visualize_shaping_reward_2d(v, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    # plt.show()


    # path = join("minigrid_basics", "function_approximation", "experiments_ablation", args.env, args.dr_mode)






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--dr_mode", type=str, default="dr_anchor")
    parser.add_argument("--lambd", help="lambda for DR", default=20.0, type=float)
    parser.add_argument("--step_size", default=1e-1, type=float, help="Starting step size")
    parser.add_argument("--barrier", help="barrier coefficient", default=0.5, type=float)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)    
    args = parser.parse_args()

    if args.dr_mode == "dr_anchor":
        args.barrier = 0

    # create env
    set_random_seed(args.seed)
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True,goal_absorbing_reward=-0.001)
    shaper = RewardShaper(env)
    visualizer = Visualizer(env)

    group_name = f"{args.env}-{args.dr_mode}-{args.step_size}-{args.barrier}"
    run_name = group_name + f"-{args.seed}"

    run = wandb.init(
        project=f"minigrid-eigen-dr-ablation-tabular-hyper",
        config=vars(args),
        group=group_name,  
        job_type="train",
        name=run_name, 
    )
    run.define_metric("train/*", step_metric="train/epoch")

    # learn
    cmap = "rainbow"
    eigenlearning_tabular(args, env)




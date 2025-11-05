import numpy as np
import random
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import subprocess
import glob

from flax import nnx
from functools import partial
import optax
import gym
import os
from os.path import join
import pickle
from copy import deepcopy
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.function_approximation.encoder import DR_Encoder

from minigrid_basics.function_approximation.learning_rules.log_objective import log_objective_step

import matplotlib.pyplot as plt
import argparse
import gin
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def norm_log_eigvec(v):
    """
    Normalize log of eigenvector such that the eigenvector has norm 1
    """
    return v - jnp.log(jnp.linalg.norm(jnp.exp(v)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms", type=str, help="Specify environment.")
    parser.add_argument("--obs_type", default="onehot", type=str, help="Type of environment observation")

    parser.add_argument("--n_epochs", type=int, default=10000)

    # hyperparams
    parser.add_argument("--step_size_start", default=1e-4, type=float, help="Starting step size")
    parser.add_argument("--step_size_end", default=3e-5, type=float, help="Ending step size")
    parser.add_argument("--grad_norm_clip", default=0.5, type=float, help="Gradient norm clipping")


    # random seed
    parser.add_argument("--seed", default=0, type=int, help="Initial random seed/key")

    args = parser.parse_args()

    # set random seed for env implemented in numpy
    set_random_seed(args.seed)

    # path = join("minigrid_basics", "function_approximation", "eigvec_learning", args.env, args.obs_type)
    # os.makedirs(path, exist_ok=True)

    # exp_dir_name = f"{args.step_size_start}-{args.step_size_end}-{args.grad_norm_clip}"
    # exp_path = join("minigrid_basics", "function_approximation", "experiments", "minigrid", args.env, args.obs_type, exp_dir_name)
    # os.makedirs(exp_path, exist_ok=True)


    # create env
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=args.seed, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    visualizer = Visualizer(env)

    # set lambd to be 1.3 for ground-truth eigenvector computation
    lambd = 1.3
    print("Lambda:", lambd)

    # load/collect dataset, use dataset with uniform sampling of all states including temrinal
    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}.pkl", "rb") as f:
        dataset = pickle.load(f)
    dataset_size = len(dataset)

    # compute ground-truth eigenvector
    shaper = RewardShaper(env)
    true_eigvec_DR = jnp.array(shaper.DR_top_log_eigenvector(lambd=lambd, normalize=False, symmetrize=True))
    true_eigvec_DR = norm_log_eigvec(true_eigvec_DR)

    # set lambd back t 1
    lambd = 1

    ### learn eigenvector
    if args.obs_type == "onehot":
        obs_dim = env.num_states
        feat_dim = 256
        step_size = 3e-4
        norm_clip = 0.5
    elif args.obs_type == "coordinates":
        obs_dim = 2
        feat_dim = 256
    elif args.obs_type == "image":
        obs_dim = 256
        feat_dim = 256
    else:
        raise NotImplementedError()

    rngs = nnx.Rngs(args.seed)

    # load test set
    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_test.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f))

    # init network
    encoder = DR_Encoder(obs_dim, feat_dim, 1, 0, 0.5, args.obs_type, rngs) 
    eigvec_old = lax.stop_gradient(jnp.squeeze(encoder(test_set)))

    n_epochs = args.n_epochs

    ss_schedule = optax.linear_schedule(
        init_value = 1e-4,
        end_value = 1e-4,
        transition_steps = n_epochs,
    )

    optimizer = nnx.Optimizer(
        encoder,
        optax.chain(
            optax.clip_by_global_norm(args.grad_norm_clip),   # clip total grad norm 
            # optax.clip(0.005),
            optax.rmsprop(ss_schedule)       # step size = 3e-4
        )
    )

    # training loop
    stats = []
    css = []
    magnitude = []
    magnitude_u = []
    norm = 0

    dataset_2 = deepcopy(dataset)

    visualize_learning = False
    obs, actions, rewards, next_obs, terminals = [jnp.array(x) for x in zip(*dataset)]

    for epoch in range(n_epochs + 1): 

        random.shuffle(dataset_2)
        obs_2, _, _, _, _ = [jnp.array(x) for x in zip(*dataset_2)]

        loss = log_objective_step(encoder, obs, rewards, next_obs, terminals, obs_2, optimizer)

       
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}: ", end="")
            eigvec_unnormalized = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
            # print(eigvec_unnormalized)
            eigvec_normalized = norm_log_eigvec(eigvec_unnormalized)
            cos_sim = eigvec_normalized @ true_eigvec_DR / (jnp.linalg.norm(eigvec_normalized) * jnp.linalg.norm(true_eigvec_DR) + 1e-8)
            css.append(cos_sim)
            magnitude.append(jnp.abs(eigvec_unnormalized).mean())

            magnitude_u.append(jnp.exp(eigvec_unnormalized * 2).sum())
            

            print(cos_sim, norm)

            print(f"  {eigvec_unnormalized}")

    print(eigvec_unnormalized)

    plt.subplot(3, 1, 1)
    plt.plot(css)
    plt.subplot(3, 1, 2)
    plt.plot(magnitude)
    plt.subplot(3, 1, 3)
    plt.plot(magnitude_u)
    plt.show()




    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(eigvec_normalized, ax=None, normalize=True, vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(true_eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1)
    plt.show()
    # plt.savefig(join(exp_path, f"{args.seed}-learned-eigvec.png"))
    plt.clf()

    mp = eigvec_myopic_policy(env, eigvec_normalized)
    visualizer.visualize_option_with_env_reward(mp)
    plt.savefig(join(exp_path, f"{args.seed}-learned-eigvec-dir.png"))
    plt.clf()

    with open(join(exp_path, f"{args.seed}-cos-sim.pkl"), "wb") as f:
        pickle.dump(css, f)

    with open(join(exp_path, f"{args.seed}-eigvec.pkl"), 'wb') as f:
        pickle.dump(eigvec_unnormalized, f)

    # stat_names = stats[0].keys()
    # n = len(stat_names)

    # for i, sn in enumerate(stat_names):
    #     plt.subplot(n + 1, 1, i + 1)
    #     stat = [s[sn] for s in stats]

    #     if "component" in sn:
    #     #     stat = np.log(stat)
    #         plt.axhline(0, color="k", linestyle="--")

    #     plt.plot(stat, label=sn)
    #     plt.legend()

    # plt.subplot(n+1, 1, n + 1)
    # plt.plot(css, label="cs")
    # plt.legend()

    # plt.show()





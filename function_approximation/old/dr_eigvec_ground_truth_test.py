"""
Experiment of learning the eigenvector from a dataset of 
transitions collected by the default policy. Defualt policy
is the uniform random policy here.

Flow:
1. Collect dataset
2. initialize DR eigvec network
3. iterate thru dataset to learn
4. Track cosine similarity and loss
5. Generate plots

"""

import numpy as np
import random
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax

from flax import nnx
from functools import partial
import optax
import gym
import os
import pickle
from copy import deepcopy
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.examples.reward_shaper import RewardShaper
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.function_approximation.encoder import DR_Encoder
import matplotlib.pyplot as plt
import argparse
import gin
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def onehot(num_states, idx):
    v = np.zeros((num_states)) * 1.
    if idx is None:
        return v
    v[idx] = 1
    return v

def collect_transitions_test(env, mode="onehot", max_iter=1000):
    """
    mode: onehot or xy or pixel, determines observation type
    """

    state_visits = np.zeros((env.num_states))

    dataset = []

    for _ in range(max_iter):

        env.reset()     # does not initialize at goal
        s = np.random.choice(env.num_states)    # randomly initialize start state
        state_visits[s] += 1

        if s in env.terminal_idx:   # when start state = terminal state
            dataset.append([onehot(env.num_states, s), 0, 0, onehot(env.num_states, None), 1.])
            continue


        x, y = env.state_to_pos[s]     
        env.unwrapped.agent_pos = [x, y]

        # action following default policy (uniform random)
        a = np.random.choice(env.num_actions)
        ns, r, done, d = env.step(a)

        if mode == "onehot":
            dataset.append((onehot(env.num_states, s), a, r, onehot(env.num_states, ns['state']), int(s in env.terminal_idx)))
        elif mode == "xy":  # coordinates
            y, x = env.agent_pos
            
            raise NotImplementedError()


    print(state_visits / state_visits.sum(), 1 / env.num_states)

    return dataset

# pass lambda, decalre static
# allo

treat_encoder_output_log = True
lambd=  1.3

@partial(nnx.jit, static_argnums=())
def allo_dr_step(encoder, optimizer, test_set, target):
    """
    encoder: network for outputting representation
    obs: current obs/state
    rewards: reward of current obs/state
    next obs: next obs/state
    terminals: whether current obs/state is a terminal state
    obs2: another set of obs/state for computing quadratic penalty term in allo
    optimizer: optimizer
    test_set: a set of all states of the environment for evaluating the learned represnetations
    true_eigvec: the true eigvec of the DR
    """

    def allo_dr_loss(encoder, test_set, target):
        
        pred = encoder(test_set)
        print(pred.shape, target.shape)
        
        allo_dr_loss = ((pred - target) ** 2).mean()
        

        return allo_dr_loss

    loss, grads = nnx.value_and_grad(allo_dr_loss)(encoder, test_set, target)
    optimizer.update(grads)

    # compute cosine similarity
    eigvec = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
    cos_sim = eigvec @ jnp.squeeze(target) / (jnp.linalg.norm(eigvec) * jnp.linalg.norm(target) + 1e-8)

    return loss, eigvec, cos_sim



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms", type=str, help="Specify environment.")
    parser.add_argument("--seed", default=0, type=int, help="Initial random seed/key")
    parser.add_argument("--obs_type", default="onehot", type=str, help="Type of environment observation")
    args = parser.parse_args()

    # set random seed for env implemented in numpy
    set_random_seed(args.seed)

    # create env
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=args.seed, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env)

    # compute ground-truth eigenvector
    shaper = RewardShaper(env)
    true_eigvec_DR = jnp.array(shaper.DR_top_log_eigenvector(lambd=lambd, normalize=False, symmetrize=True))
    print(true_eigvec_DR)

    target = true_eigvec_DR.reshape(-1, 1)

    visualizer = Visualizer(env)

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
    # TODO: use argparse for encoder params
    encoder = DR_Encoder(obs_dim, feat_dim, 1, 0, 1, args.obs_type, rngs)
    
    # gradient clipping
    optimizer = nnx.Optimizer(
        encoder,
        optax.chain(
            optax.clip_by_global_norm(0.5),   # clip total grad norm 
            # optax.clip(0.005),
            optax.adamw(3e-4)
        )
    )

    # training loop
    css = []
    for _ in range(5000): #6000

        

        loss, eigvec, cos_sim = allo_dr_step(encoder, optimizer, test_set, target)
        print(loss, cos_sim)
        css.append(cos_sim)

        if jnp.isnan(cos_sim):
            print(lax.stop_gradient(jnp.squeeze(encoder(test_set))))
            quit()

    print(eigvec)
    print(true_eigvec_DR)
        

    

    plt.subplot(1, 2, 1)
    # visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(true_eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1)
    plt.show()

    plt.plot(css)
    plt.show()





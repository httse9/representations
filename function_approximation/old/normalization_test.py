"""


"""

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
import matplotlib.pyplot as plt
import argparse
import gin
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def create_env(env_id, formulation, debug=True):
    formulation = int(formulation) # just in case formulation is string
    """
    For discussion on formulations, see comment at the top.
    """
    env = gym.make(env_id)
    if formulation == 1:
        env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=False)
    elif formulation == 2:
        env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True, goal_absorbing_reward=-1e-3)
        """
        Note that when we set goal_absorbing, we are not changng the underlying MDP formulation.
        We are just changing how we formulate the DR, which is part of the solution.
        The terminal state in the MDP formulation is still not absorbing, and still has reward 0.
        But we formulate it as absorbing and having a minor negative reward in the computation of the DR.
        """

    if debug:
        print("*****************")
        print(f"Formulation {formulation}")
        print(f"Terminal state reward in constructed reward vector: {env.rewards[env.terminal_idx[0]]}")
        print(f"Terminal state transition probs in constructed P: {env.transition_probs[env.terminal_idx[0], :, env.terminal_idx[0]]}")
        

        env.reset()
        x, y = env.state_to_pos[env.terminal_idx[0]]
        env.unwrapped.agent_pos = [x, y]
        ns, r, done, d = env.step(0)
        print(f"Actual reward received at terminal state: {r}")

        print("*****************")

    # verify that envs have all 0 terminal reward. (done)
    # verify that R & P depends on goal_absorbing correctly. (done)
    # verify that when interacting with environment, still receives 0 reward at terminal state. (done)
    # verify that DR (using RewardShaper) is computed using R & P (done)

    return env

@nnx.jit
def true_normalization(encoder, test_set, target, optimizer):

    def compute_loss(encoder, test_set, target):
        eigvec = encoder(test_set)

        # 

        return ((eigvec - target) ** 2).mean(), eigvec

    (loss, eigvec), grads = nnx.value_and_grad(compute_loss, has_aux=True)(encoder, test_set, target)
    optimizer.update(grads)

    norm = jnp.exp(eigvec * 2).sum()

    return loss, eigvec, norm


class Normalizer:

    def __init__(self, encoder, test_set):
        self.encoder = encoder
        self.test_set = test_set

    def normalize(self):

        optimizer = nnx.Optimizer(
            self.encoder,
            optax.adam(1e-3)
        )

        eigvec = lax.stop_gradient(self.encoder(self.test_set))
        target = eigvec - jnp.log(jnp.linalg.norm(jnp.exp(eigvec)))
        # print("Target", target.squeeze())

        step = 0
        while True:
            step += 1
            loss, eigvec, norm = true_normalization(self.encoder, self.test_set, target, optimizer)
            # print(loss, norm)

            if jnp.abs(target - eigvec).mean() < 0.01 or step >= 3000:
            # if (loss < 0.1 and jnp.abs(norm - 1) < 0.05) or step >= 2000:
                # print("Final norm:",norm)
                # print(eigvec.squeeze())
                break

        # print("Total steps", step)

        return norm


def initialize(encoder, test_set):

    n_states = test_set.shape[0]
    target = jnp.array(np.random.normal(loc=-50, scale=5, size=n_states)).reshape(-1, 1)
    # print("Target")
    # print(target.squeeze())
    optimizer = nnx.Optimizer(
        encoder,
        optax.adam(1e-3)
    )

    def compute_loss(encoder, test_set, target):

        return ((encoder(test_set) - target) ** 2).mean()

    loss, grads = nnx.value_and_grad(compute_loss, has_aux=False)(encoder, test_set, target)
    optimizer.update(grads)

    return loss



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dayan_2", type=str, help="Specify environment.")
    parser.add_argument("--seed", default=0, type=int, help="Initial random seed/key")
    parser.add_argument("--obs_type", default="coordinates", type=str, help="Type of environment observation")
    args = parser.parse_args()

    # create env
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = create_env(env_id, 1)
    shaper = RewardShaper(env)
    visualizer = Visualizer(env)


    np.random.seed(args.seed)


    # create encoder
    obs_dim = 2
    feat_dim = 256
    rngs = nnx.Rngs(args.seed)
    encoder = DR_Encoder(obs_dim, feat_dim, 1, 0, 0.5, args.obs_type, rngs) 

    # load test set
    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_test.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f))


    # initialize to a very bad vector
    loss = 10
    for _ in range(200):
        loss = initialize(encoder, test_set)
        # print(loss)

    eigvec = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
    print("*************")
    print("INITIALIZATION")
    print(eigvec)
    print(jnp.exp(2 * eigvec).sum())


    print("**************")
    print("TARGET")
    eigvec_true_normalized = eigvec - jnp.log(jnp.linalg.norm(jnp.exp(eigvec)))
    print(eigvec_true_normalized)
    print(jnp.exp(2 * eigvec_true_normalized).sum())

    


    # normalization
    normalizer = Normalizer(encoder, test_set)
    normalizer.normalize()

    # after normalization
    eigvec = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
    print("**************")
    print("NORMALIZED")
    print(eigvec)
    print(jnp.exp(2 * eigvec).sum())



    plt.plot(eigvec_true_normalized, label="target")
    plt.plot(eigvec, label="normalized")
    plt.legend()
    plt.show()
    


    """
    WRONG!!!

    Currently, since the target is always moving, I'm in fact not normalizing the 
    vector by shifting everything upwards. I'm just squashing everything.
    Maybe need to reintroduce bias in the encoder's final layer?
    """
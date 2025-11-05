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
def condition(encoder, obs, rewards, next_obs, terminals, obs2, optimizer, test_set, true_eigvec):
    """
    Make network predict same for all observations/states?
    """
    def condition_loss(encoder, obs, rewards, next_obs, terminals, obs2):
        log_phi = encoder(obs)
        return ((log_phi - 0.5) ** 2).mean()

    loss, grads = nnx.value_and_grad(condition_loss)(encoder, obs, rewards, next_obs, terminals, obs2)
    optimizer.update(grads)

    return loss


@partial(nnx.jit, static_argnums=())
def allo_dr_step(encoder, obs, rewards, next_obs, terminals, obs2, optimizer, test_set, true_eigvec):
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

    def allo_dr_loss(encoder, obs, rewards, next_obs, terminals, obs2):
        """
        We make some simplifications here:
        1. Use a fixed barrier coefficient large enough, no need to gradient ascent
        2. Simply add all losses together without scaling each loss
        """

        n_states = len(test_set)

        # To ensure that the network outputs positive values, we treat the network output
        # as the log of the DR. So take exponential here to get eigenvector
        if treat_encoder_output_log:
            log_phi = encoder(obs)
            log_next_phi = encoder(next_obs)
            log_phi_2 = encoder(obs2)

            phi = jnp.exp(log_phi)
            next_phi = jnp.exp(log_next_phi)
            phi_2 = jnp.exp(log_phi_2)
        else:
            phi = encoder(obs)
            next_phi = encoder(next_obs)
            phi_2 = encoder(obs2)
        
        

        barrier_coefficients = encoder.barrier_coefs
        duals = encoder.duals

        ### the GDO objective for DR
        # graph_loss = (phi ** 2) * jnp.expand_dims(jnp.exp(-rewards / lambd), 1)
        # graph_loss -= jnp.expand_dims(1 - terminals, 1) * phi * next_phi
        # graph_loss = graph_loss.mean()

        ### Natural gradient
        # graph_loss = lax.stop_gradient(2 * jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.exp(log_next_phi - log_phi)) * log_phi
        # graph_loss += lax.stop_gradient(-jnp.exp(log_phi - log_next_phi)) * log_next_phi * jnp.expand_dims(1 - terminals, 1)

        # equivalent implementation, + not compute loss for terminal state
        # graph_loss = lax.stop_gradient(jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.exp(log_next_phi - log_phi)) * log_phi * 2 * (jnp.expand_dims(1 - terminals, 1) + jnp.expand_dims(terminals, 1) * 0.5)
        # jnp.expand_dims(1 - terminals, 1) + jnp.expand_dims(terminals, 1) * 0.5

        


        # asymmetric
        # graph_loss += jnp.expand_dims(1 - terminals, 1) * lax.stop_gradient(-jnp.exp(log_phi - log_next_phi)) * log_next_phi
        # graph_loss = graph_loss.mean()


        # glps_r = jnp.exp(-rewards / lambd).reshape(n_states, -1).mean(1)
        # graph_loss_per_state = lax.stop_gradient(jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.exp(log_next_phi - log_phi)).reshape(n_states, -1).mean(1)

        

        true_loss = (phi ** 2) * jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.expand_dims(1 - terminals, 1) * phi * next_phi
        true_loss = true_loss.mean()

        ## dual loss
        # compute inner product of state/obs representations
        n = phi.shape[0]
        # inner_product_matrix_1 = jnp.einsum(
        #     'ij,ik->jk', log_phi, lax.stop_gradient(log_phi)) / n
        # inner_product_matrix_2 = jnp.einsum(
        #     'ij,ik->jk', log_phi_2, lax.stop_gradient(log_phi_2)) / n
        # # subtract identity
        # error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(encoder.eig_dim))
        # error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(encoder.eig_dim))
        # # avg cause why not
        # error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)

        # dual_loss_pos = (lax.stop_gradient(jnp.asarray(duals)) * error_matrix).sum()
        # dual_loss_neg = - (duals * lax.stop_gradient(error_matrix)).sum()


        ### dual loss nat grad version (assume eig_dim = 1)
        # dual_loss_pos = 0       # sufficient for one hot to converge, version below doesn't work)
        # dual_loss_neg = 0
        # dual_loss_pos = (lax.stop_gradient(jnp.asarray(duals)) * 2 * log_phi).mean()
        # dual_loss_neg = - (duals * lax.stop_gradient(jnp.exp(2 * log_phi) - 1)).mean()

        ### quadratic loss
        # quadratic_error_matrix = error_matrix_1 * error_matrix_2
        # barrier_loss_pos = barrier_coefficients * quadratic_error_matrix.sum()

        ### quadratic loss nat grad version (assume eig_dim = 1)
        # barrier_loss_pos = log_phi * lax.stop_gradient(phi.T @ phi / n - n_states) + log_phi_2 * lax.stop_gradient(phi_2.T @ phi_2 / n - n_states)
        # barrier_loss_pos = log_phi * lax.stop_gradient(jnp.exp(2 * log_phi_2) - 1) + log_phi_2 * lax.stop_gradient(jnp.exp(2 * log_phi) - 1)


        # barrier_loss_pos = log_phi * lax.stop_gradient(phi_2.T @ phi_2 / n - 1) + log_phi_2 * lax.stop_gradient(phi.T @ phi / n - 1)

        # barrier_loss_pos = barrier_loss_pos.mean() * barrier_coefficients


        ### mean loss: ensure output of neural net has 0 mean
        # mean_loss = 0.01 * (log_phi.mean() ** 2)        # works for fourrooms coordinates
        # mean_loss = -log_phi.mean() 

        ### Total loss
        # allo_dr_loss = graph_loss + dual_loss_pos + dual_loss_neg + barrier_loss_pos # + mean_loss

        allo_dr_loss = lax.stop_gradient(jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.exp(log_next_phi - log_phi) + 2 * barrier_coefficients * (jnp.exp(2 * log_phi_2) - 1)) * log_phi
        allo_dr_loss = allo_dr_loss.mean()

        individual_losses = lax.stop_gradient(dict(
            # graph_loss=graph_loss,
            # dual_loss_pos=dual_loss_pos,
            # dual_loss_neg=dual_loss_neg,
            # barrier_loss_pos=barrier_loss_pos,
            # graph_loss_per_state=graph_loss_per_state,
            # glps_r=glps_r,
            true_loss=true_loss
        ))

        return allo_dr_loss, individual_losses

    (loss, individual_losses), grads = nnx.value_and_grad(allo_dr_loss, has_aux=True)(encoder, obs, rewards, next_obs, terminals, obs2)
    optimizer.update(grads)

    # compute cosine similarity
    if treat_encoder_output_log:
        eigvec = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
    else:
        eigvec = lax.stop_gradient(jnp.squeeze(jnp.log(jnp.abs(encoder(test_set)))))
    cos_sim = eigvec @ true_eigvec / (jnp.linalg.norm(eigvec) * jnp.linalg.norm(true_eigvec) + 1e-8)

    return loss, individual_losses, eigvec, cos_sim



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

    # load/collect dataset
    # dataset_size = 500
    # dataset = collect_transitions_test(env, max_iter=dataset_size)
    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}.pkl", "rb") as f:
        dataset = pickle.load(f)
    dataset_size = len(dataset)

    # compute ground-truth eigenvector
    shaper = RewardShaper(env)
    true_eigvec_DR = jnp.array(shaper.DR_top_log_eigenvector(lambd=lambd, normalize=False, symmetrize=True))
    print(true_eigvec_DR)

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
    encoder = DR_Encoder(obs_dim, feat_dim, 1, 0, 0.5, args.obs_type, rngs)     # b = 1.
    # # initialize to have all positive weights
    # for layer in encoder.eig_linear.layers:
    #     if hasattr(layer, "kernel"):
    #         # print(layer)
    #         layer.kernel.value = jnp.abs(layer.kernel.value) / 10
    #         assert (layer.kernel.value > 0).all()

    # plain optimizer
    # optimizer = nnx.Optimizer(encoder, optax.adam(3e-4))

    eigvec_init = lax.stop_gradient(encoder(test_set).squeeze())

    # gradient clipping
    # schedule = optax.exponential_decay(
    # init_value=1e-4,       # initial LR
    # transition_steps=100,  # how many steps before multiplying by decay_rate
    # decay_rate=0.99,       # multiplicative decay factor
    # end_value=1e-6         # (optional) minimum LR
    # )

    optimizer = nnx.Optimizer(
        encoder,
        optax.chain(
            optax.clip_by_global_norm(0.5),   # clip total grad norm 
            # optax.clip(0.005),
            optax.adam(1e-4)       # step size = 3e-4
        )
    )

    # training loop
    dataset2 = deepcopy(dataset)
    random.shuffle(dataset2)
    batch_size = dataset_size 
    css = []
    true_losses = []
    e0 = []
    et = []

    # for _ in range(100):
    #     obs, actions, rewards, next_obs, terminals = [jnp.array(x) for x in zip(*dataset)]
    #     loss = condition(encoder, obs, rewards, next_obs, terminals, obs, optimizer, test_set, true_eigvec_DR)
    #     print(loss)

    for _ in range(5000): #6000

        # shuffle dataset
        # random.shuffle(dataset)
        random.shuffle(dataset2)

        for i in range(0, dataset_size, batch_size):
            batch = dataset[i:i+batch_size]
            obs, actions, rewards, next_obs, terminals = [jnp.array(x) for x in zip(*batch)]

            batch2 = dataset2[i:i+batch_size]           # !!! huge bug here, I used dataset instead of dataset 2
            obs2, _, _, _, _ = [jnp.array(x) for x in zip(*batch2)]

            loss, individual_losses, eigvec, cos_sim = allo_dr_step(encoder, obs, rewards, next_obs, terminals, obs2, optimizer, test_set, true_eigvec_DR)
            print(cos_sim, individual_losses['true_loss'])
            # print(eigvec)
            css.append(cos_sim)
            true_losses.append(individual_losses['true_loss'])


            # glps = individual_losses['graph_loss_per_state']
            # print(glps)

            
            # plt.subplot(2,3, 1)
            # visualizer.visualize_shaping_reward_2d(eigvec_init, ax=None, normalize=True, vmin=0, vmax=1)
            # plt.subplot(2, 3, 2)
            # visualizer.visualize_shaping_reward_2d(glps, ax=None, normalize=True, vmin=0, vmax=1)
            # plt.subplot(2, 3, 3)
            # visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
            # plt.subplot(2, 3, 4)
            # visualizer.visualize_shaping_reward_2d(individual_losses['glps_r'], ax=None, normalize=True, vmin=0, vmax=1)
            

            # plt.show()
            

            # if individual_losses["graph_loss"] > 1000:
            #     print(eigvec)
            #     visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
            #     plt.show()

            if jnp.isnan(cos_sim):
                print(lax.stop_gradient(jnp.squeeze(encoder(test_set))))
                quit()

            eigvec_init = eigvec
    
    print(eigvec)
    print(true_eigvec_DR)
        

    

    plt.subplot(1, 2, 1)
    # visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(true_eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1)
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(css, label="cs")
    plt.subplot(2, 1, 2)
    plt.plot(jnp.log(jnp.array(true_losses)), label="tl")
    plt.legend()
    plt.show()





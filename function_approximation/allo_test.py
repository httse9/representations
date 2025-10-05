from minigrid_basics.examples.ROD_cycle import RODCycle
import os
import numpy as np
from flint import arb_mat, ctx
from itertools import islice
from minigrid_basics.function_approximation.encoder import Encoder
from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.examples.reward_shaper import RewardShaper


ctx.dps = 100   # important

# testing imports
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import subprocess
import glob
import pickle
from minigrid_basics.examples.rep_utils import *

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from functools import partial
import random


def onehot(num_states, idx):
    v = np.zeros((num_states)) * 1.
    if idx is None:
        return v
    v[idx] = 1
    return v
    

def collect_transitions_test(env, mode="state_num", max_iter=1000):
    """
    mode: state_num or xy or pixel, determines observation type
    """

    dataset = []

    s = env.reset()
    n_steps = 0
    while n_steps < max_iter:

        s = env.reset()

        # action following default policy (uniform random)
        a = np.random.choice(env.num_actions)

        ns, r, done, d = env.step(a)

        if mode == "state_num":
            dataset.append((onehot(env.num_states, s['state']), a, r, onehot(env.num_states, ns['state']), 0.))
            n_steps += 1
        else:
            raise NotImplementedError()


        if done:
            if d["terminated"]:
                # for _ in range(100):
                dataset.append((onehot(env.num_states, ns['state']), 0, 0, onehot(env.num_states, None), 1.))
                # n_steps += 100

            s = env.reset()
        else:
            s = ns

    return dataset

 
# @partial(nnx.jit, static_argnums=(5,))
# def jitted_update_step(encoder, encoder_optimizer, observations, next_observations, observations_2, eig_dim, aux):

#     max_barrier_coefs, step_size_duals, min_duals, max_duals, barrier_scale = aux
    
#     def encoder_loss(encoder, observations, next_observations, observations_2):
        
#         phi = jnp.exp(encoder(observations))
#         phi_2 = jnp.exp(encoder(observations_2))

#         next_phi = jnp.exp(encoder(next_observations))

#         graph_loss = 0.5 * ((phi - next_phi)**2).mean(0).sum()

#         barrier_coefficients = encoder.barrier_coefs.clip(0, max_barrier_coefs)
#         duals = encoder.duals #.clip(min_duals, max_duals)
                
#         # Compute errors
#         n = phi.shape[0]
#         inner_product_matrix_1 = jnp.einsum(
#             'ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
#         inner_product_matrix_2 = jnp.einsum(
#             'ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n

#         error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(eig_dim))
#         error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(eig_dim))

#         error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)

#         dual_loss_pos = (jax.lax.stop_gradient(jnp.asarray(duals)) * error_matrix).sum()

#         dual_loss_neg = - step_size_duals * (duals * jax.lax.stop_gradient(error_matrix)).sum()

#         quadratic_error_matrix = error_matrix_1 * error_matrix_2
#         barrier_loss_pos = jax.lax.stop_gradient(barrier_coefficients[0,0]) * quadratic_error_matrix.sum()
        
#         quadratic_error = jnp.clip(quadratic_error_matrix, 0, None).mean()
#         barrier_loss_neg = -barrier_coefficients[0,0] * jax.lax.stop_gradient(quadratic_error)

#         # Total loss
#         allo_loss = dual_loss_pos + barrier_loss_pos + graph_loss + dual_loss_neg + (barrier_loss_neg * barrier_scale)
        
#         indiv_losses = {
#             'dual_loss_pos' : dual_loss_pos,
#             'barrier_loss_pos' : barrier_loss_pos,
#             'graph_loss' : graph_loss, 
#             'dual_loss_neg' : dual_loss_neg,
#             'barrier_loss_neg' : barrier_loss_neg,
#             'allo_loss' : allo_loss
#             }
        
#         return allo_loss, indiv_losses

#     (allo_loss, indiv_losses), grads = nnx.value_and_grad(encoder_loss, has_aux=True)(encoder, observations, next_observations, observations_2)
    
#     encoder_optimizer.update(grads)
    
#     return indiv_losses


@partial(nnx.jit, static_argnums=(7,))
def jitted_update_step_DR(encoder, encoder_optimizer, observations, rewards, next_observations, terminals, observations_2, eig_dim, aux):

    """
    Modify graph_loss for DR
    """
    # aux = jnp.array([1e5, 1., -100, 100, 1e3])
    max_barrier_coefs, step_size_duals, min_duals, max_duals, barrier_scale = aux
    
    def encoder_loss(encoder, observations, rewards, next_observations, terminals, observations_2):
        
        phi = encoder(observations)
        phi_2 = encoder(observations_2)

        next_phi = encoder(next_observations)




        # DR graph loss

        ### 0th theoretically correct objective
        graph_loss = (phi ** 2) * jnp.expand_dims(jnp.exp(-rewards), 1)
        graph_loss -= jnp.expand_dims(1 - terminals, 1) * phi * next_phi

        ### 1st combination that works
        # graph_loss = (phi ** 2) * jnp.expand_dims(-rewards, 1)
        # graph_loss -= jnp.expand_dims(1 - terminals, 1) * phi * next_phi
        # Note: adding jnp.exp() for rewards causes failure.


        ### Hypothesis: the penalize reward part too strong. the second term not enough to pull updates
        # everything graviates towards 0.
        ### Proposed solution: Add a weight term for the second term
        # graph_loss = (phi ** 2) * jnp.expand_dims(jnp.exp(-rewards), 1)
        # graph_loss -= 5 * jnp.expand_dims(1 - terminals, 1) * phi * next_phi
        ### Result: works like a charm! w=100 too strong. No need log visualization
        # Note: Seems like more epoch is bette (well, hasn't converged..). Larger step size ruins things
        # 1000 epoch looks quite nice. 
        # Now things seem to be flipped though. Terminal state is not standing out. Change to reward 1? Not much change
        # # Maybe it's because terminal state is not sampled enough? It's only there when episode terminates. Not a sampled state. Change?
        # # Quick check: just manually add more. Doesn't help.
        # Problem: While reward-aware, does not guide towards goal. How to solve?
            # It does guide towards areas with low risk of stepping into low rewards, e.g., top right room in 4rooms_2

        ### What if, isntead of up-weighting the second term. Use lambda to down-weight the 1st term
        # graph_loss = (phi ** 2) * jnp.expand_dims(jnp.exp(-rewards / 1.3), 1)
        # graph_loss -= jnp.expand_dims(1 - terminals, 1) * phi * next_phi
        ### Result: loss seem more well-behaved (above 0 all the time). But cannot get good results
        # / 5 looks pretty cool. Less than 5 nothing looks nice
        # Need to log to see good results (?)

        # print(graph_loss.shape)
        # quit()

        graph_loss = graph_loss.mean()


        barrier_coefficients = encoder.barrier_coefs.clip(0, max_barrier_coefs)
        duals = encoder.duals #.clip(min_duals, max_duals)
                
        # Compute errors
        n = phi.shape[0]
        inner_product_matrix_1 = jnp.einsum(
            'ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
        inner_product_matrix_2 = jnp.einsum(
            'ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n

        error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(eig_dim))
        error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(eig_dim))

        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)

        dual_loss_pos = (jax.lax.stop_gradient(jnp.asarray(duals)) * error_matrix).sum()

        dual_loss_neg = - step_size_duals * (duals * jax.lax.stop_gradient(error_matrix)).sum()

        quadratic_error_matrix = error_matrix_1 * error_matrix_2
        barrier_loss_pos = jax.lax.stop_gradient(barrier_coefficients[0,0]) * quadratic_error_matrix.sum()
        
        quadratic_error = jnp.clip(quadratic_error_matrix, 0, None).mean()
        barrier_loss_neg = -barrier_coefficients[0,0] * jax.lax.stop_gradient(quadratic_error)

        # Total loss
        allo_loss = dual_loss_pos + barrier_loss_pos + graph_loss + dual_loss_neg + (barrier_loss_neg * barrier_scale)
        
        indiv_losses = {
            'dual_loss_pos' : dual_loss_pos,
            'barrier_loss_pos' : barrier_loss_pos,
            'graph_loss' : graph_loss, 
            'dual_loss_neg' : dual_loss_neg,
            'barrier_loss_neg' : barrier_loss_neg,
            'allo_loss' : allo_loss
            }
        
        return allo_loss, indiv_losses

    (allo_loss, indiv_losses), grads = nnx.value_and_grad(encoder_loss, has_aux=True)(encoder, observations, rewards, next_observations, terminals, observations_2)
    
    encoder_optimizer.update(grads)
    
    return indiv_losses


def train_allo_test(env,):

    allo_losses = []
    # full_dataset = collect_transitions_test(env, max_iter=10000)
    with open("minigrid_basics/function_approximation/static_dataset/fourrooms_2_onehot.pkl", "rb") as f:
        dataset = pickle.load(f)

    eig_dim = 1
    encoder = Encoder(obs_dim = env.num_states, feat_dim = 256, eig_dim = eig_dim, 
            duals_initial_val = 0, barrier_initial_val = 0,
            obs_type = "one_hot", rngs = nnx.Rngs(seed))

    optimizer = nnx.Optimizer(encoder, optax.adam(0.0003))

    key = jax.random.PRNGKey(42)

    for i in range(5000):

        random.shuffle(dataset)
        states, actions, rewards, next_states, terminals = [jnp.array(x) for x in zip(*dataset)]

        if i == 0:
            sm = states.mean(0)

        key, subkey = jax.random.split(key)
        states_2 = states[jax.random.permutation(subkey ,states.shape[0])]
        
        aux = jnp.array([1e5, 1., -100, 100, 1e3])

        # print(jitted_update_step(encoder, optimizer, states, next_states, states_2, 1, aux)['graph_loss'])
        print(jitted_update_step_DR(encoder, optimizer, states, rewards, next_states, terminals, states_2, 1, aux)['graph_loss'])


    # print(allo_losses)

    encoder_test(env, encoder, sm)

def encoder_test(env, encoder, sm):
    
    eigvec = []
    for i in range(env.num_states):
        s = onehot(env.num_states, i)
        e = encoder(s)
        eigvec.append(e)

    eigvec = np.abs(np.array(eigvec))

    print(eigvec)
    # print(eigvec[28])
    # eigvec[28] = np.median(eigvec)
    

    visualizer = Visualizer(env)
    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(sm, ax=None, normalize=True, vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(np.log(eigvec), ax=None, normalize=True, vmin=0, vmax=1)
    plt.show()

def eigvec_myopic_policy(env, eigvec):
    """
    Get the myopic (hill-climbing policy) for current eigenvector
    """
    termination = np.zeros((env.num_states))
    policy = np.zeros((env.num_states))

    for s in range(env.num_states):

        # handle unvisited state / terminal state
        if s in env.terminal_idx:
            termination[s] = 1
            continue

        # for visited states:
        pos = env.state_to_pos[s]  # (x, y): x-th col, y-th row
        value = eigvec[s]  # init value
        myopic_a = -1

        for a, dir_vec in enumerate(np.array([
            [1, 0], # right
            [0, 1], # down
            [-1, 0],    # left
            [0, -1],    # up
        ])):
            neighbor_pos = pos + dir_vec
            neighbor_state = env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * env.width]
            
            # if neighbor state exists (not wall) 
            # and neighor state has been visited
            # and has higher eigenvector value
            # go to that neighbor state
            if neighbor_state >= 0 and eigvec[neighbor_state] > value:
                value = eigvec[neighbor_state]
                myopic_a = a

        if myopic_a == -1:
            # no better neighbor, terminate
            termination[s] = 1
        else:
            policy[s] = myopic_a

    myopic_policy = dict(termination=termination, policy=policy)
    return myopic_policy


def plot_3d(env, v):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    value_map = construct_value_pred_map(env, v, contain_goal_value=True).T
    value_map[np.isinf(value_map)] = np.median(value_map[~np.isinf(value_map)])

    # get rid of walls
    value_map = value_map[1:-1, 1:-1]

    value_map = np.rot90(value_map, k=1)

    x, y = value_map.shape
    x, y = np.meshgrid(range(x), range(y))
    ax.plot_surface(x, y, value_map, cmap=cm.Reds, linewidth=0, antialiased=False)
    # ax.title.set_text("SR")

    # plt.show()

if __name__ == "__main__":
    
    env_name = "fourrooms_2"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()


    # create env
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make(env_id, seed=seed, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env)


    # visualizer = Visualizer(env)
    # shaper = RewardShaper(env)

    # visualizer.visualize_env()
    # plt.show()

    # SR = shaper.compute_DR()
    # plt.imshow(SR)
    # plt.show()
    # eigvec_DR = np.log(SR[31])

    # plt.subplot(1, 2,1 )
    # eigvec_DR = shaper.DR_top_log_eigenvector()
    # visualizer.visualize_shaping_reward_2d(eigvec_DR)
    # plt.subplot(1, 2, 2)
    # eigvec_DR = shaper.SR_top_eigenvector()
    # eigvec_DR = shaper.shaping_reward_transform_using_terminal_state(eigvec_DR) * -1
    # visualizer.visualize_shaping_reward_2d(eigvec_DR)
    # plt.show()
    # myopic_policy = eigvec_myopic_policy(env, eigvec_DR )
    # visualizer.visualize_option_with_env_reward(myopic_policy)
    # plt.show()

    
    # plot_3d(env, eigvec_DR)
    # plt.axis("off")
    # plt.tight_layout()
    # ax = plt.gca()
    # ax.view_init(azim=-88)
    # plt.show()
    

    # print(env.terminal_idx[0])

    # generate dataset
    # dataset = collect_transitions_test(env, max_iter=200)
    # print(dataset)
    # print(len(dataset))

    # ##
    train_allo_test(env,)



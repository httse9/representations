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

def norm_log_eigvec(v):
    """
    Normalize log of eigenvector such that the eigenvector has norm 1
    """
    return v - jnp.log(jnp.linalg.norm(jnp.exp(v)))

def insert_zero(v, idx):
    return jnp.insert(v, idx, 0.)

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
        log_phi = encoder(obs)
        log_next_phi = encoder(next_obs) * jnp.expand_dims(1 - terminals, 1)        # anchor terminal states
        log_phi_2 = encoder(obs2)

        phi = jnp.exp(log_phi)
        next_phi = jnp.exp(log_next_phi)
        phi_2 = jnp.exp(log_phi_2)
        
        barrier_coefficients = encoder.barrier_coefs
        duals = encoder.duals

        reward_component = jnp.expand_dims(jnp.exp(-rewards / lambd), 1)
        graph_component = lax.stop_gradient(-jnp.exp((log_next_phi - log_phi) ))
        norm_component = 0# lax.stop_gradient(2 * barrier_coefficients * (jnp.exp(2 * log_phi_2).mean() - 1))

        component_sum = reward_component + graph_component + norm_component

        allo_dr_loss = (component_sum)  * log_phi
        allo_dr_loss = allo_dr_loss.mean()

        reward_component_per_state = reward_component.reshape(-1, 4).mean(-1)
        graph_component_per_state = graph_component.reshape(-1, 4).mean(-1)
        norm_component_per_state = norm_component
        component_sum_per_state = reward_component_per_state + graph_component_per_state + norm_component_per_state

        # stats
        phi_normalized = lax.stop_gradient(phi / jnp.linalg.norm(phi))
        next_phi_normalized = lax.stop_gradient(next_phi / jnp.linalg.norm(next_phi))

        true_loss = (phi ** 2) * jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.expand_dims(1 - terminals, 1) * phi * next_phi
        true_loss = true_loss.mean()

        true_loss_normalized = (phi_normalized ** 2)  * jnp.expand_dims(jnp.exp(-rewards / lambd), 1) - jnp.expand_dims(1 - terminals, 1) * phi_normalized * next_phi_normalized
        true_loss_normalized = true_loss_normalized.mean()


        individual_losses = lax.stop_gradient(dict(
            true_loss=true_loss,
            true_loss_normalized=true_loss_normalized,
            reward_component=reward_component_per_state,
            graph_component=graph_component_per_state,
            norm_component=norm_component_per_state,
            component_sum=component_sum_per_state
        ))

        return allo_dr_loss, individual_losses

    (loss, individual_losses), grads = nnx.value_and_grad(allo_dr_loss, has_aux=True)(encoder, obs, rewards, next_obs, terminals, obs2)
    optimizer.update(grads)

    # compute cosine similarity
    eigvec_unnormalized = lax.stop_gradient(jnp.squeeze(encoder(test_set)))
    eigvec_unnormalized = eigvec_unnormalized.at[env.terminal_idx[0]].set(0)      # anchor terminal state

    # normalize eigvec
    eigvec = norm_log_eigvec(eigvec_unnormalized)

    cos_sim = eigvec @ true_eigvec / (jnp.linalg.norm(eigvec) * jnp.linalg.norm(true_eigvec) + 1e-8)

    return loss, individual_losses, eigvec_unnormalized, cos_sim



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

    path = join("minigrid_basics", "function_approximation", "eigvec_learning", args.env, args.obs_type)
    os.makedirs(path, exist_ok=True)

    exp_dir_name = f"{args.step_size_start}-{args.step_size_end}-{args.grad_norm_clip}"
    exp_path = join("minigrid_basics", "function_approximation", "experiments", "minigrid", args.env, args.obs_type, exp_dir_name)
    os.makedirs(exp_path, exist_ok=True)


    # create env
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=args.seed, no_goal=False, no_start=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    visualizer = Visualizer(env)

    # set lambd to be max reward magnitude
    lambd = jnp.abs(env.rewards).max()
    print("Lambda:", lambd)

    # load/collect dataset
    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_2.pkl", "rb") as f:
        dataset = pickle.load(f)
    dataset_size = len(dataset)

    # compute ground-truth eigenvector
    shaper = RewardShaper(env)
    true_eigvec_DR = jnp.array(shaper.DR_top_log_eigenvector(lambd=lambd, normalize=False, symmetrize=True))
    true_eigvec_DR = norm_log_eigvec(true_eigvec_DR)

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
        init_value = args.step_size_start,
        end_value = args.step_size_end,
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
    dataset2 = deepcopy(dataset)
    random.shuffle(dataset2)
    batch_size = dataset_size 
    stats = []
    css = []
    eigvec_magnitude = [jnp.abs(eigvec_old).mean()]

    visualize_learning = False

    for epoch in range(n_epochs + 1): 

        # shuffle dataset
        random.shuffle(dataset2)

        for i in range(0, dataset_size, batch_size):
            batch = dataset[i:i+batch_size]
            obs, actions, rewards, next_obs, terminals = [jnp.array(x) for x in zip(*batch)]

            batch2 = dataset2[i:i+batch_size]   
            obs2, _, _, _, _ = [jnp.array(x) for x in zip(*batch2)]

            loss, individual_losses, eigvec, cos_sim = allo_dr_step(encoder, obs, rewards, next_obs, terminals, obs2, optimizer, test_set, true_eigvec_DR)
            
            if epoch % 50 == 0:
                print(f"{epoch}:", cos_sim, individual_losses['true_loss'], individual_losses['true_loss_normalized'])
                print("  ", individual_losses['reward_component'].mean(), individual_losses['graph_component'].mean())

            if visualize_learning:
                plt.figure(figsize=(9, 6))
                plt.subplot(2, 3, 1)
                visualizer.visualize_shaping_reward_2d(eigvec_old, ax=None, normalize=True, vmin=0, vmax=1)
                plt.title(f"eigvec pre-update {eigvec_old.min():.2f}/{eigvec_old.max():.2f}")

                plt.subplot(2, 3, 3)
                plt.title(f"eigvec post-update {eigvec.min():.2f}/{eigvec.max():.2f}")
                visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1)
                
                plt.subplot(2, 3, 4)
                rc = insert_zero(individual_losses["reward_component"], env.terminal_idx[0])
                plt.title(f"reward comp. {rc.min():.2f}/{rc.max():.2f}")
                visualizer.visualize_vec(rc)

                plt.subplot(2,3, 5)
                gc = insert_zero(individual_losses['graph_component'], env.terminal_idx[0])
                plt.title(f"graph comp. {gc.min():.2f}/{gc.max():.2f}")
                visualizer.visualize_shaping_reward_2d(-gc, cmap="Blues")

                plt.subplot(2, 3, 6)
                plt.title(f"norm comp. {individual_losses['norm_component']:.2f}")
                visualizer.visualize_vec([individual_losses['norm_component']] * env.num_states)

                plt.subplot(2, 3, 2)
                cs = insert_zero(individual_losses['component_sum'], env.terminal_idx[0])
                plt.title(f"update {cs.min():.2f}/{cs.max():.2f}")
                visualizer.visualize_vec(cs)

                plt.suptitle(f"Epoch {epoch}")
                plt.tight_layout()
                plt.savefig(join(path, f"{epoch}.png"))
                plt.close()
            
            stats.append(individual_losses)
            css.append(cos_sim)
            eigvec_magnitude.append(jnp.abs(eigvec).mean())

            eigvec_old = eigvec

            if jnp.isnan(cos_sim):
                print("!!!Overflow!!!")
                print(lax.stop_gradient(jnp.squeeze(encoder(test_set))))
                break

        if jnp.isnan(cos_sim):
            break

    
    eigvec_normalized = norm_log_eigvec(eigvec)
    
    print(eigvec_normalized)
    print(true_eigvec_DR)
        
    if visualize_learning:
        # save video
        os.chdir(path)
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', f'%d.png', '-r', '30','-pix_fmt', 'yuv420p', 
            '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            '-y', f'eigvec_learning.mp4'
        ])

        for file_name in  glob.glob("*.png"):
            os.remove(file_name)

    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(eigvec_normalized, ax=None, normalize=True, vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(true_eigvec_DR, ax=None, normalize=True, vmin=0, vmax=1)
    plt.savefig(join(exp_path, f"{args.seed}-learned-eigvec.png"))
    plt.clf()

    mp = eigvec_myopic_policy(env, eigvec_normalized)
    visualizer.visualize_option_with_env_reward(mp)
    plt.savefig(join(exp_path, f"{args.seed}-learned-eigvec-dir.png"))
    plt.clf()

    with open(join(exp_path, f"{args.seed}-cos-sim.pkl"), "wb") as f:
        pickle.dump(css, f)

    with open(join(exp_path, f"{args.seed}-eigvec.pkl"), 'wb') as f:
        pickle.dump(eigvec, f)

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





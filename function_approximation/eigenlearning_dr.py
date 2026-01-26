import numpy as np
import jax.numpy as jnp
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
from os.path import join
import subprocess
import glob
from minigrid_basics.function_approximation.eigenlearner_FA import *
import wandb

learners = {
    "dr_anchor": DRLearner,
    "dr_norm": DR_NORM_Learner,
    "dr_gdo": DRGDOLearner,
    "dr_gdo_log": DRGDOLOGLearner
}

def eigenlearning(args, env):

    ### load dataset
    if args.dataset == "static":
        dataset, test_set = load_static_dataset(args)
    elif args.dataset == "real":
        dataset, test_set = load_dataset(args)

     ### for visualizing eigvec
    visualizer = Visualizer(env)
    cmap = "rainbow"  

    ### init learner
    learner = learners[args.dr_mode](env, dataset, test_set, args)
    learner.init_learn()

    ### learn
    learner.learn()

    eigvec = learner.eigvec()

    ### save plots
    # plt.axhline(1.0, linestyle="--", color='k')
    # plt.plot(learner.cos_sims)
    # plt.xlabel("Time Steps (x100)")
    # plt.ylabel("Cosine Similarity")
    # plt.tight_layout()
    # plt.savefig(join(plot_path, f"{run_name}_cossim.png"))
    # plt.clf()

    # # plt.axhline(env.num_states, linestyle="--", color="k")
    # plt.plot(learner.norms)
    # plt.xlabel("Time Steps (x100)")
    # plt.ylabel("Eigvec Norm")
    # plt.tight_layout()
    # plt.savefig(join(plot_path, f"{run_name}_eigvec-norm.png"))
    # plt.clf()

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    visualizer.visualize_shaping_reward_2d(learner.true_eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.subplot(1, 3, 2)
    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.subplot(1, 3, 3)
    pi = eigvec_myopic_policy(env, eigvec)
    visualizer.visualize_option_with_env_reward(pi)
    plt.tight_layout()
    plt.savefig(join(plot_path, f"{run_name}_eigvec.png"))
    wandb.log({"eigvec_vis": wandb.Image(plt)})
    plt.clf()

    ### save raw data
    data = dict(
        cos_sims=learner.cos_sims,
        norms=learner.norms,
        true_eigvec=learner.true_eigvec,
        eigvec=eigvec
    )
    with open(join(data_path, f"{run_name}.pkl"), "wb") as f:
        pickle.dump(data, f)


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

### Ground-truth small dataset (for testing)
def load_static_dataset(args):
    if args.dr_mode == "dr_anchor":
        with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_2.pkl", "rb") as f:
            dataset = pickle.load(f)
    elif args.dr_mode in ["dr_norm", "dr_gdo", "dr_gdo_log"]:
        with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}.pkl", "rb") as f:
            dataset = pickle.load(f)

    with open(f"minigrid_basics/function_approximation/static_dataset/{args.env}_{args.obs_type}_test.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f))

    return dataset, test_set

### Actual dataset used in paper
def load_dataset(args):

    if args.dr_mode in ["dr_norm", "dr_gdo", "dr_gdo_log"]:
        with open(f"minigrid_basics/function_approximation/dataset/{args.env}_norm_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)[args.obs_type]

    else:
        with open(f"minigrid_basics/function_approximation/dataset/{args.env}_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)[args.obs_type]

    with open(f"minigrid_basics/function_approximation/dataset/{args.env}_testset.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f)[args.obs_type])

    return dataset, test_set



def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--obs_type", default="onehot", type=str)
    parser.add_argument("--dataset", default="static", type=str)

    parser.add_argument("--dr_mode", type=str, default="dr_anchor")
    parser.add_argument("--optimizer", default="rmsprop", type=str)
    
    parser.add_argument("--lambd", help="lambda for DR", default=20.0, type=float)
    parser.add_argument("--step_size_start", default=1e-5, type=float, help="Starting step size")
    parser.add_argument("--step_size_end", default=None, type=float, help="Ending step size. Default: None, meaning equal to start")
    parser.add_argument("--grad_norm_clip", default=0.5, type=float, help="Ending step size")

    parser.add_argument("--n_epochs", type=int, default=400, help="Number of passes thru dataset")
    parser.add_argument("--batch_size", default=2000, help="Batch size", type=int)
    parser.add_argument("--log_interval", default=100, type=int, help="interval to compute cosine similarity")

    parser.add_argument("--eig_dim", type=int, default=1, help="How many dimension of laplacian representation to learn")
    parser.add_argument("--barrier", type=float, default=0.5, help="Barrier coefficient b. Not used for anchor")

    parser.add_argument("--save_model", action="store_true", help="Whether to save trained network.")
    
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.step_size_end is None:
        args.step_size_end = args.step_size_start

    if "anchor" in args.dr_mode:
        args.barrier = 0

    # create env
    set_random_seed(args.seed)
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True, goal_absorbing_reward=-0.001)


    # create dir for saving results
    path = join("minigrid_basics", "function_approximation", f"experiments_{args.dr_mode}_{args.dataset}", args.env, args.obs_type)
    plot_path = join(path, "plots")
    data_path = join(path, "data")
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    group_name = f"{args.lambd}-{args.step_size_start}-{args.step_size_end}-{args.batch_size}-{args.barrier}-{args.optimizer}-{args.dr_mode}-{args.grad_norm_clip}"
    run_name = group_name + f"-{args.seed}"

    run = wandb.init(
        project=f"testing-minigrid-eigen-dr-{args.env}-{args.obs_type}-{args.dataset}",
        config=vars(args),
        group=group_name,  
        job_type="train",
        name=run_name, 
    )
    run.define_metric("train/*", step_metric="train/epoch")

    # learn
    eigenlearning(args, env)

    run.finish()



    
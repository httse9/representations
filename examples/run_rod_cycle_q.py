import os
from os.path import join
from minigrid_basics.examples.ROD_SR_q_learning import ROD_SR_Q
from minigrid_basics.examples.ROD_DR_q_learning import ROD_DR_Q
import argparse
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import numpy as np
import pickle
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the ROD cycle.")

    parser.add_argument("--env", type=str, default="fourrooms", help="Environment name")
    parser.add_argument("--representation", default="SR", help="Which representation: [SR/DR]")

    # collect sample step
    parser.add_argument("--n_steps", type=int, default=100, help="Number of steps in each episode")
    parser.add_argument("--p_option", type=float, default=0.05, help="Probability of choosing an option")

    # learning the representation
    parser.add_argument("--dataset_size", type=int, default=None, help="Size of dataset. Default is None, using all data")
    parser.add_argument("--learn_rep_iteration", type=int, default=10, help="Number of iterations for learning representation")
    parser.add_argument("--representation_step_size", type=float, default=0.1, help="Step size for representation learning")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for eigenoption/SR")

    # eigenoption
    parser.add_argument("--num_options", type=int, default=None, help="Number of options to keep. Default is None, keeping all options")
    parser.add_argument("--eigenoption_step_size", type=float, default=0.1, help="Step size for learning eigenoptions")

    # DR specific
    # parser.add_argument('--lambd')

    # plotting
    parser.add_argument("--plot", action="store_true", help="Whether to plot results (True/False)")


    # seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")



    return parser.parse_args()

def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  

def create_env(env_name, seed):
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id, seed=seed, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    return env


if __name__ == "__main__":

    n_iterations = {
        "dayan": 50,
        "dayan_2": 50,
        "fourrooms": 50,      
        "fourrooms_2": 50,
        "gridroom": 120,
        "gridroom_2": 120,
        "gridmaze": 120,
        "gridmaze_2": 120
    }
    
    args = vars(get_args())
    env_name = args.pop("env")
    seed = args.pop("seed")
    representation = args.pop("representation")


    args['env'] = create_env(env_name, seed)
    set_random_seed(seed)

    if representation == "SR":
        rod_cycle = ROD_SR_Q(**args)
    elif representation == "DR":
        rod_cycle = ROD_DR_Q(**args)
    else:
        raise ValueError(f"Representation {representation} not recognized.")

    rewards, visit_percentage = rod_cycle.rod_cycle(n_iterations=n_iterations[env_name])

    ### save results
    path = join("minigrid_basics", "experiments", "rod_q", env_name, representation)
    os.makedirs(path, exist_ok=True)

    keys = ["p_option", "dataset_size", "learn_rep_iteration", "representation_step_size", "num_options"]
    values = [str(args[key]) for key in keys]
    filename = '-'.join(values) + f"-{seed}.pkl"

    
    data_dict = dict(
        Q_performance=rod_cycle.Q_performance,
    )

    with open(join(path, filename), "wb") as f:
        pickle.dump(data_dict, f)




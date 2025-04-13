import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter, Colors
from itertools import product

from minigrid_basics.examples.visualizer import Visualizer
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import gin
import gym


rod_directory = join("minigrid_basics", "experiments", "rod")

# build the file name given the hyperparameters
def construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed):
    values = [p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options]
    values = [str(v) for v in values]
    filename = '-'.join(values) + f"-{seed}.pkl"
    return filename

# read data given env_name, representation, and hyperparameters
def read_data(env_name, representation, p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed=10):
    path = join(rod_directory, env_name, representation)

    visit = []
    num_successful_seeds = 0

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)

        try:
            with open(join(path, filename), "rb") as f:
                data = pickle.load(f)

            visit.append(data['all_iteration_state_visits'])
            num_successful_seeds += 1
        except:
            pass

    return num_successful_seeds, np.array(visit)

## hyperparameters
SR_best_hyperparameters = [
    [0.05, 100, 1, 0.1, 1],
    [0.1, 100, 10, 0.1, 1],
    [0.1, 100, 10, 0.1, 1],
    [0.1, 100, 100, 0.01, 1]
]


DR_best_hyperparameters = [
    [0.05, 100, 1, 0.1, 1],
    [0.1, 100, 10, 0.03, 1],
    [0.1, 100, 1, 0.03, 1],
    [0.1, 100, 1, 0.03, 1]
]


if __name__ == "__main__":

    envs = ["dayan_2", "fourrooms_2",  "gridroom_2", "gridmaze_2",]

    env_labels = [
        "Modified Grid Task",
        "Four Rooms",
        "Modified Grid Room",
        "Modified Grid Maze", 
    ]

    plotter = Plotter()

    ### Figure 5
    # visualization of cumulative visit of SR and DR in four low-reward environments

    fix, axs = plt.subplots(2, 4, figsize=(12, 6))


    for env_name, ax, SR_hyper, DR_hyper in zip(envs, axs.T, SR_best_hyperparameters, DR_best_hyperparameters):

        # SR visits
        _, SR_visits = read_data(env_name, "SR", *SR_hyper)
        # cumulative visit over all iterations
        SR_visits = SR_visits.sum(1)
        # average over seeds
        SR_visits = SR_visits.mean(0)
        
        # DR visits
        ss, DR_visits = read_data(env_name, "DR", *DR_hyper)
        DR_visits = DR_visits.sum(1).mean(0)

        SR_visits = np.log(SR_visits)
        DR_visits = np.log(DR_visits)

        vmin = min(SR_visits.min(), DR_visits.min())
        vmax = max(SR_visits.max(), DR_visits.max())
        
        ### Visualize
        # make env
        gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
        env_id = maxent_mon_minigrid.register_environment()
        env = gym.make(env_id, seed=42)
        env = maxent_mdp_wrapper.MDPWrapper(env, )

        # visualize
        visualizer = Visualizer(env)
        visualizer.visualize_shaping_reward_2d(SR_visits, ax=ax[0], normalize=False, vmin=vmin, vmax=vmax)
        visualizer.visualize_shaping_reward_2d(DR_visits, ax=ax[1], normalize=False, vmin=vmin, vmax=vmax)

    
    plt.savefig(f"minigrid_basics/plots/rod_Figure_5.png", dpi=300)
        
    
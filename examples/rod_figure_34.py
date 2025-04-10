import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter, Colors
from itertools import product

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

    all_rewards = []
    all_visit_percentage = []
    num_successful_seeds = 0

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)

        try:
            with open(join(path, filename), "rb") as f:
                data = pickle.load(f)

            all_rewards.append(data['rewards'])
            all_visit_percentage.append(data['visit_percentage'])
            num_successful_seeds += 1
        except:
            pass

    return num_successful_seeds < seed, np.array(all_rewards), np.array(all_visit_percentage)



SR_best_hyperparameters = [
    [0.05, 100, 1, 0.1, 1],
    [0.05, 100, 1, 0.1, 1],
    [0.1, 100, 10, 0.1, 1],
    [0.1, 100, 10, 0.1, 1],
    [0.1, 100, 10, 0.1, 1],
    [0.1, 100, 10, 0.1, 1],
    [0.1, 100, 100, 0.01, 1],
    [0.1, 100, 100, 0.01, 1]
]

DR_best_hyperparameters = [
    [0.01, 100, 10, 0.01, 1],
    [0.05, 100, 1, 0.1, 1],
    [0.1, 100, 10, 0.03, 1],
    [0.1, 100, 10, 0.03, 1],
    [0.05, 100, 1, 0.01, 1],
    [0.1, 100, 1, 0.03, 1],
    [0.1, 100, 1, 0.01, 1],
    [0.1, 100, 1, 0.03, 1]
]

if __name__ == "__main__":

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2",]

    env_labels = [
        "Modified Grid Task",
        "Four Rooms",
        "Modified Grid Room",
        "Modified Grid Maze", 
    ]
    env_labels = [x for x in env_labels for _ in range(2)]

    

    # # read random walk data
    # for env_name in envs:
    #     path = join(rod_directory, env_name)
        
    #     rw_r = []
    #     rw_p = []
    #     for s in range(1, 11):
    #         with open(join(path, f"random_walk_{s}.pkl"), "rb") as f:
    #             data = pickle.load(f)
    #             rw_r.append(data['rewards'])
    #             rw_p.append(data['visit_percentage'])

    #      = np.array(rw_p)
    #     = np.array(rw_r)

    plotter = Plotter()

    ### Figure 3

    fig, axs = plt.subplots(2, 4, figsize=(21, 8))
    axs = axs.T.flatten()
    for env_name, env_label, ax, SR_hyper, DR_hyper in zip(envs, env_labels, axs, SR_best_hyperparameters, DR_best_hyperparameters):

        _, r_sr, p_sr = read_data(env_name, "SR", *SR_hyper)
        _, r_dr, p_dr = read_data(env_name, "DR", *DR_hyper)

        plotter.index = 0
        x = np.array(range(p_sr.shape[1])) * 1.0
        x /= 1000
        plotter.plot_data(ax, x, p_sr, conf_level=0.95, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 1
        plotter.plot_data(ax, x, p_dr, plot_conf_int=False, plot_all_seeds=True)

        if "2" in env_name:
            x_label = "Thousand Steps"
            title = None
        else:
            x_label = None
            title = env_label
        if "dayan" in  env_name:
            y_label = "State Visit Percentage"
        else:
            y_label = None

        if env_name == "dayan":
            ax.annotate("Without Low-Reward Region", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
            
        elif env_name == "dayan_2":
            ax.annotate("With Low-Reward Region", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
        

        plotter.finalize_plot(ax, title=title, xlabel=x_label, ylabel=y_label, axis_setting=(None, None, 0, None))

    plt.savefig(f"minigrid_basics/plots/rod_Figure_3.png", dpi=300)
    plt.close()

    ### Figure 3

    fig, axs = plt.subplots(1, 4, figsize=(21, 4))
    axs = axs.T.flatten()
    idx = ["2" in e for e in envs]
    envs = [x for x, m in zip(envs, idx) if m]
    env_labels =[x for x, m in zip(env_labels, idx) if m]
    SR_best_hyperparameters = [x for x, m in zip(SR_best_hyperparameters, idx) if m]
    DR_best_hyperparameters = [x for x, m in zip(DR_best_hyperparameters, idx) if m]

    for env_name, env_label, ax, SR_hyper, DR_hyper in zip(envs, env_labels, axs, SR_best_hyperparameters, DR_best_hyperparameters):
        
        _, r_sr, p_sr = read_data(env_name, "SR", *SR_hyper)
        _, r_dr, p_dr = read_data(env_name, "DR", *DR_hyper)

        r_sr = np.cumsum(r_sr, axis=1)
        r_dr = np.cumsum(r_dr, axis=1)

        plotter.index = 0
        x = np.array(range(r_sr.shape[1])) * 1.0
        x /= 1000
        plotter.plot_data(ax, x, r_sr, conf_level=0.95, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 1
        plotter.plot_data(ax, x, r_dr, plot_conf_int=False, plot_all_seeds=True)

    
        if "dayan" in  env_name:
            y_label = "Cumulative Rewards"
        else:
            y_label = None
        

        plotter.finalize_plot(ax, title=env_label, xlabel="Thousand Steps", ylabel=y_label, )

    plt.savefig(f"minigrid_basics/plots/rod_Figure_4.png", dpi=300)



    # keys = []
    # for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
    #     hyper_strings = [str(v) for v in list(hyper)]
    #     hypername = '-'.join(hyper_strings)
    #     keys.append(hypername)


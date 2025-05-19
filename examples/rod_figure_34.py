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
    [0.1, 100, 100, 0.01, 1],
    # [0.05, 100, 100, 0.01, 1],    # gridroom_25
    # [0.05, 100, 100, 0.01, 1]     # gridmaze_29
]

DR_best_hyperparameters = [
    [0.05, 100, 10, 0.01, 1],
    [0.05, 100, 1, 0.1, 1],
    [0.05, 100, 1, 0.1, 1],
    [0.1, 100, 10, 0.01, 1],
    [0.1, 100, 1, 0.01, 1],
    [0.1, 100, 1, 0.03, 1],
    [0.1, 100, 1, 0.01, 1],
    [0.1, 100, 1, 0.01, 1],
    # [0.05, 100, 1, 0.03, 1],
    # [0.1, 100, 1, 0.03, 1]
]


if __name__ == "__main__":

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2"]#, "gridroom_25", "gridmaze_29"]

    env_labels = [
        "Grid Task",
        "Four Rooms",
        "Grid Room",
        "Grid Maze", 
    ]
    

    

    # read random walk data
    def random_walk_data(env_name):
        path = join(rod_directory, env_name)
        
        rw_r = []
        rw_p = []
        for s in range(1, 11):
            with open(join(path, f"random_walk_{s}.pkl"), "rb") as f:
                data = pickle.load(f)
                rw_r.append(data['rewards'])
                rw_p.append(data['visit_percentage'])

        return np.array(rw_p), np.array(rw_r)

    plotter = Plotter()

    ### Figure 3 (state visitation for envs with no low reward region)
    idx = ["2" not in e for e in envs]
    envs_fig_3 = [x for x, m in zip(envs, idx) if m]
    SR_best = [x for x, m in zip(SR_best_hyperparameters, idx) if m]
    DR_best = [x for x, m in zip(DR_best_hyperparameters, idx) if m]
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.5))
    axs = axs.flatten()
    for env_name, env_label, ax, SR_hyper, DR_hyper in zip(envs_fig_3, env_labels, axs, SR_best, DR_best):

        _, r_sr, p_sr = read_data(env_name, "SR", *SR_hyper)
        _, r_dr, p_dr = read_data(env_name, "DR", *DR_hyper)
        p_rw, r_rw = random_walk_data(env_name)

        plotter.index = 0
        x = np.array(range(p_sr.shape[1])) * 1.0
        x /= 1000
        plotter.plot_data(ax, x, p_sr, conf_level=0.95, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 2
        plotter.plot_data(ax, x, p_rw, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 1
        plotter.plot_data(ax, x, p_dr, plot_conf_int=False, plot_all_seeds=True)

        if "dayan" in  env_name:
            y_label = "State Visitation %"
            plotter.index = 0
            plotter.draw_text(ax, -0.2, 0.95, "CEO")
            plotter.index = 1
            plotter.draw_text(ax, 3, 0.85, "RACE")
            plotter.index = 2
            plotter.draw_text(ax, 1.2, 0.8, "RW")
        else:
            y_label = None

            

        plotter.finalize_plot(ax, title=env_label, xlabel="Steps ($×10^3$)", ylabel=y_label, axis_setting=(None, None, 0, None))

    plt.savefig(f"minigrid_basics/plots/rod_Figure_3.png", dpi=300)
    plt.close()

    ### Figure 4

    fig, axs = plt.subplots(2, 4, figsize=(13, 5))
    axs = axs.T
    idx = ["2" in e for e in envs]#
    envs = [x for x, m in zip(envs, idx) if m]
    # env_labels = [x for x in env_labels for _ in range(2)]
    SR_best_hyperparameters = [x for x, m in zip(SR_best_hyperparameters, idx) if m]
    DR_best_hyperparameters = [x for x, m in zip(DR_best_hyperparameters, idx) if m]

    for env_name, env_label, ax, SR_hyper, DR_hyper in zip(envs, env_labels, axs, SR_best_hyperparameters, DR_best_hyperparameters):
        
        
        _, r_sr, p_sr = read_data(env_name, "SR", *SR_hyper)
        _, r_dr, p_dr = read_data(env_name, "DR", *DR_hyper)
        p_rw, r_rw = random_walk_data(env_name)

        r_sr = np.cumsum(r_sr, axis=1)
        r_dr = np.cumsum(r_dr, axis=1)
        r_rw = np.cumsum(r_rw, axis=1)

        # plot visitation top row
        plotter.index = 0
        x = np.array(range(p_sr.shape[1])) * 1.0
        x /= 1000
        plotter.plot_data(ax[0], x, p_sr, conf_level=0.95, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 2
        plotter.plot_data(ax[0], x, p_rw, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 1
        plotter.plot_data(ax[0], x, p_dr, plot_conf_int=False, plot_all_seeds=True)

        

        if "dayan" in  env_name:
            plotter.index = 0
            plotter.draw_text(ax[0], -0.2, 0.95, "CEO")
            plotter.index = 1
            plotter.draw_text(ax[0], 3, 0.9, "RACE")
            plotter.index = 2
            plotter.draw_text(ax[0], 1.2, 0.8, "RW")


        y_label = "State Visitation %" if "dayan" in env_name else None

        plotter.finalize_plot(ax[0], title=env_label, xlabel=None, ylabel=y_label,)


        # plot cumulative reward bottom row
        plotter.index = 0
        x = np.array(range(r_sr.shape[1])) * 1.0
        x /= 1000
        plotter.plot_data(ax[1], x, r_sr / 1000, conf_level=0.95, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 2
        plotter.plot_data(ax[1], x, r_rw / 1000, plot_conf_int=False, plot_all_seeds=True)

        plotter.index = 1
        plotter.plot_data(ax[1], x, r_dr / 1000, plot_conf_int=False, plot_all_seeds=True)

        


        # y ticks
        min_return = min(r_sr.min(), r_dr.min(), r_rw.min()) / 1000
        min_return = int(np.ceil(min_return))
        interval = 5
        tick_positions = range(0, min_return - 1, min_return // interval)
        ax[1].set_yticks(tick_positions)

        y_label = "Cumulative Rewards ($×10^3$)" if "dayan" in env_name else None

        plotter.finalize_plot(ax[1], title=None, xlabel="Steps ($×10^3$)", ylabel=y_label)

    plt.savefig(f"minigrid_basics/plots/rod_Figure_4.png", dpi=300)



    # keys = []
    # for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
    #     hyper_strings = [str(v) for v in list(hyper)]
    #     hypername = '-'.join(hyper_strings)
    #     keys.append(hypername)


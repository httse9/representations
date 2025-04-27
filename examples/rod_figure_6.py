import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter, Colors
from itertools import product

rod_directory = join("minigrid_basics", "experiments", "rod_q")

# build the file name given the hyperparameters
def construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed):
    values = [p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options]
    values = [str(v) for v in values]
    filename = '-'.join(values) + f"-{seed}.pkl"
    return filename

# read data given env_name, representation, and hyperparameters
def read_data(env_name, representation, p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed=50):
    path = join(rod_directory, env_name, representation)

    q_performance = []
    num_successful_seeds = 0

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)

        try:
            with open(join(path, filename), "rb") as f:
                data = pickle.load(f)

            q_performance.append(data['Q_performance'])
            num_successful_seeds += 1
        except:
            pass

    return num_successful_seeds < seed, np.array(q_performance)

def construct_q_filename(init, eps, stepsize, seed):
    values = [init, eps, stepsize]
    values = [str(v) for v in values]
    filename = '-'.join(values) + f"-{seed}.pkl"
    return filename

def read_q_data(env, init, eps, stepsize):

    fpath = join(rod_directory, env, "qlearning")
    q_performance = []
    for s in range(1, 51):  

        filename = construct_q_filename(init, eps, stepsize, s)
        with open(join(fpath, filename), "rb") as f:
            data = pickle.load(f)
        q_performance.append(data['Q_performance'])

    return np.array(q_performance)



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
    [0.05, 100, 10, 0.01, 1],
    [0.05, 100, 1, 0.1, 1],
    [0.05, 100, 1, 0.1, 1],
    [0.1, 100, 10, 0.01, 1],
    [0.1, 100, 1, 0.01, 1],
    [0.1, 100, 1, 0.03, 1],
    [0.1, 100, 1, 0.01, 1],
    [0.1, 100, 1, 0.01, 1]
]

qlearning_best_hyperparameters = [
    [-10, 0.2, 0.01],
    [-10, 0.15, 0.01],
    [-10, 0.2, 1.0],
    [-10, 0.15, 1.0],
    [-10, 0.2, 0.3],
    [-10, 0.2, 0.3],
    [-10, 0.2, 0.3],
    [0, 0.2, 1.0]
]

if __name__ == "__main__":

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2",]

    env_labels = [
        "Modified Grid Task",
        "Four Rooms",
        "Modified Grid Room",
        "Modified Grid Maze", 
        "",
        "",
        "",
        ""
    ]
    



    plotter = Plotter()

    ### Figure 3 (state visitation for envs with no low reward region)

    fig, axs = plt.subplots(2, 4, figsize=(13, 5))
    axs = axs.T.flatten()
    for env_name, env_label, ax, SR_hyper, DR_hyper, q_hyper in zip(envs, env_labels, axs, SR_best_hyperparameters, DR_best_hyperparameters, qlearning_best_hyperparameters):
        print(env_name)
        _, q_sr = read_data(env_name, "SR_nt", *SR_hyper)
        _, q_dr = read_data(env_name, "DR", *DR_hyper)
        q_q = read_q_data(env_name, *q_hyper)

        # print(q_sr[:, -1].argmin())

        max_performance = max(q_sr.mean(0).max(), q_dr.mean(0).max(), q_q.mean(0).max())

        
        x = np.array(range(q_sr.shape[1])) * 1.0

        plotter.index = 2
        plotter.plot_data(ax, x, q_q, plot_conf_int=True)

        plotter.index = 0
        plotter.plot_data(ax, x, q_sr, plot_conf_int=True, plot_all_seeds=False)

        plotter.index = 1
        plotter.plot_data(ax, x, q_dr, plot_conf_int=True, plot_all_seeds=False)


        if "dayan" in  env_name:
            y_label = "Return"
        else:
            y_label = None

        if env_name == "dayan":
                plotter.index = 0
                plotter.draw_text(ax, 7, -23, "CEO(Q)")
                plotter.index = 1
                plotter.draw_text(ax, 35, -15, "RACE(Q)")
                plotter.index = 2
                plotter.draw_text(ax, 25, -50, "QL")

        if "2" in env_name:
            x_label = "Number of Iterations"
        else:
            x_label = None
            

        plotter.finalize_plot(ax, title=None, xlabel=x_label, ylabel=y_label, axis_setting=(None, None, -100, None))

    plt.savefig(f"minigrid_basics/plots/rod_Figure_6.png", dpi=300)
    plt.close()

    
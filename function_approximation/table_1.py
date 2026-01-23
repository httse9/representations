"""
Generate Numbers for table

Also plot learning curve
"""


import numpy as np
import pickle
from os.path import join
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter
from copy import deepcopy

def compute_bootstrap_conf_int(x, n_bootstrap=1000, conf_level=0.95):
    """
    Given array of x, compute bootstrap conf int
    """
    n = len(x)
    idxs = np.random.randint(0, n, size=(n_bootstrap, n))
    x_resampled = x[idxs]
    sample_means = x_resampled.mean(1)

    alpha = (1 - conf_level) / 2
    lower_p = 100 * alpha
    upper_p = 100 * (1 - alpha)
    lower = np.percentile(sample_means, lower_p)
    upper = np.percentile(sample_means, upper_p)

    return lower, upper


def compute_N_opt(ret):
    """
    Take an array of returns. Find time step till stably achieve final return
    """
    final = ret[-1]

    for i in reversed(range(len(ret))):
        if ret[i] != final:
            break

    return i + 1, final

def load_all_data(opath):

    N_opts = []
    N_visits = []
    final_returns = []
    for seed in range(1, 11):
        spath = join(opath, f"{seed}.pkl")

        with open(spath, "rb") as f:
            data = pickle.load(f)
        
        N_visits.append(data['vlr'])
        N_opt, final = compute_N_opt(data['ret'])
        N_opts.append(N_opt)
        final_returns.append(final)
        

    return np.array(N_opts).astype(float), np.array(N_visits).astype(float), np.array(final_returns)

def figure_1():
    """
    Loop over env
    Load learning data
    use plotter to plot
    Compute 95% bootstrap confidence interval
    """

    # fix random seed for bootstrap conf int
    np.random.seed(0)
    
    plotter = Plotter()
    envs = ["dayan_2", "fourrooms_2", "gridmaze_2", "gridroom_2"]
    modes = ["none-none", "SR-SR", "DR-onehot", "DR-coordinates", "DR-image"]
    stats = ["N_opt", "N_visit", "final return"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2))

    for env, ax in zip(envs, axes):

        print(env)

        path = join("minigrid_basics/function_approximation", "experiments_rs", env)

        for mode in modes:
            opath = join(path, mode)

            N_opts, N_visits, final_returns = load_all_data(opath)

            N_opts /= 1000
            N_visits /= 100

            print("  >>", mode, f"{final_returns.mean():.2f}")

            for name, x in zip(stats, [N_opts]):#, N_visits, final_returns]):

                mean = x.mean()
                lower, upper = compute_bootstrap_conf_int(x)

                print("      ", name, f"{mean:.2f}  ({lower - mean:.2f}, {upper - mean:.2f})")


    axis_settings = [
        [0, 2, 0.8, 1.05],
        [0, 3, 0.8, 1.05],
        [0, 30, 0.8, 1.05],
        [0, 30, 0.8, 1.05],
    ]
    env_names = [
        "Grid Task",
        "Four Rooms",
        "Grid Maze",
        "Grid Room"
    ]
    for env, ax, axset in zip(env_names, axes, axis_settings):
        ylabel = "Cos. Sim." if env == "Grid Task" else None
        plotter.finalize_plot(ax, env, r"Epochs ($\times 10^2$)", ylabel, axset)

    ax0 = axes[0]
    plotter.index = 0
    plotter.draw_text(ax0, 0.07, 1.01, "one-hot", 10)
    plotter.index += 1
    plotter.draw_text(ax0, 0.1, 0.85, "coord.", 10)
    plotter.index += 1
    plotter.draw_text(ax0, 1, 0.94, "pixels", 10)    
    
    plt.savefig(join("minigrid_basics/function_approximation/plots/figure_1.png"), dpi=300)
                

if __name__ == "__main__":
    figure_1()

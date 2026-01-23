"""
Plot cosine similarity for 4 environments
- dayan_2
- fourrooms_2
- gridmaze_2
- gridroom_2
for 3 obs types
- onehot
- coordinates
- image

Since 10 seeds, used bootstrapped conf int
"""


import numpy as np
import pickle
from os.path import join
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter
from copy import deepcopy

def load_all_data(opath):

    data = []
    for seed in range(1, 11):
        spath = join(opath, f"20.0-1e-05-1e-05-2000-0-rmsprop-dr_anchor-0.5-{seed}.pkl")

        with open(spath, "rb") as f:
            cos_sim = pickle.load(f)['cos_sims']
        data.append(cos_sim)

    return np.array(data)

def figure_1():
    """
    Loop over env
    Load learning data
    use plotter to plot
    Compute 95% bootstrap confidence interval
    """
    
    plotter = Plotter()
    envs = ["dayan_2", "fourrooms_2", "gridmaze_2", "gridroom_2"]
    obs_types = ["onehot", "coordinates", "image"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2))

    for env, ax in zip(envs, axes):

        path = join("minigrid_basics/function_approximation", "experiments_dr_anchor_real", env)
        plotter.index = 0

        for obs_type in obs_types:
            opath = join(path, obs_type, "data")

            cos_sims = load_all_data(opath)
            x = np.array(range(cos_sims.shape[1])) / 100

            plotter.plot_data(ax, x, cos_sims, boostrap=True)
            plotter.index += 1


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

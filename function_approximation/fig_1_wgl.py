import numpy as np
import pickle
from os.path import join
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter
from copy import deepcopy

def get_filename(hypersetting, seed):
    lambd, step_size_start, step_size_end, batch_size, barrier = hypersetting
    run_name = f"{lambd}-{step_size_start}-{step_size_end}-{batch_size}-{barrier}-{seed}.pkl"

    return run_name

def load_data(data_path, env, obs_type, hypersetting):
    
    cosine_similarity = []

    for seed in range(1, 11):
        f = get_filename(hypersetting, seed)


        with open(join(data_path, env, obs_type, "data", f), "rb") as f:
            data = pickle.load(f)

        cs = data['cos_sims']
        # cs = np.array(cs)
        cosine_similarity.append(cs)

    return np.array(cosine_similarity)

def figure_1():
    plotter = Plotter()

    """
    lambda we use 20.

    large env batch size change to 500

    for gridmaze_2 image step size change to 0.0001
    """

    hyper = [20.0, 0.0003, 0.0003, 250, 0.5]


    data_path = join("minigrid_basics", "function_approximation", "experiments_test")


    fig, axes = plt.subplots(2, 4, figsize=(10, 3))

    envs = ["dayan", "fourrooms", "gridroom", "gridmaze", "dayan_2", "fourrooms_2", "gridroom_2", "gridmaze_2"]
    titles = ["Grid Task", "Four Rooms", "Grid Room", "Grid Maze"] + [None] * 4

    xlim = np.array([200, 300, 2000, 3000, 200, 300, 2000, 4000]) / 100

    for env, ax, xl, title in zip(envs, axes.flatten(), xlim, titles):

        for obs_type in ["onehot", "coordinates", "image"]:

            hypersetting = deepcopy(hyper)
            if "grid" in env:
                hypersetting[3] = 500
            if env == "gridmaze_2" and obs_type == "image":
                hypersetting[1] = 0.0001
                hypersetting[2] = 0.0001

            cosine_similarity = load_data(data_path, env, obs_type, hypersetting)
            cosine_similarity = cosine_similarity.mean(-1)  # average over 10 eigenvectors

            plotter.plot_data(ax, np.array(range(cosine_similarity.shape[1])) / 100, cosine_similarity)

            if env == "dayan":
                if obs_type == "onehot":
                    plotter.draw_text(ax, 0.3, 0.4, "one-hot", 10)
                elif obs_type == "image":
                    plotter.draw_text(ax, 1.5, 0.75, "image", 10)
                elif obs_type == "coordinates":
                    plotter.draw_text(ax, 0.1, 1.0, "coord.", 10)

            plotter.index += 1

        xlabel = r"Gradient Steps ($\times 10^2$)" if "2" in env else None
        ylabel = "Cos. Sim." if "dayan" in env else None
        axis_setting = [0.05, xl, 0., 1.05]

        


        plotter.finalize_plot(ax, title=title, xlabel=xlabel, ylabel=ylabel, axis_setting=axis_setting)
        plotter.reset()



    plt.savefig(join("minigrid_basics", "function_approximation", "plots", "minigrid_cs.png"), dpi=300)


if __name__ == "__main__":
    figure_1()
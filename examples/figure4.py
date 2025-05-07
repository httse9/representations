import matplotlib.pyplot as plt
import numpy as np
from minigrid_basics.examples.plotter import Plotter
from os.path import join
from os import listdir
import pickle

# def read_data(env, mode):
#     data_path = join("minigrid_basics", "experiments", "reward_shaping", env, mode)
#     # print(data_path)

#     return_list = []

#     for file in listdir(data_path):
#         # skip non data files
#         if "pkl" not in file:
#             continue

#         with open(join(data_path, file), "rb") as f:
#             data = pickle.load(f)
#             return_list.append(data['ret'])

#     return_list = np.array(return_list)
#     return data['t'], return_list

def read_data(env, mode, r_aux_w, step_size):

    data_path = join("minigrid_basics", "experiments", "reward_shaping", env, mode)
    # print(data_path)

    return_list = []

    for seed in range(51, 101):
        filename = f"{r_aux_w}-{step_size}-{seed}.pkl"

        with open(join(data_path, filename), "rb") as f:
            data = pickle.load(f)
            return_list.append(data['ret'])

    return_list = np.array(return_list)
    return data['t'], return_list


if __name__ == "__main__":

    plotter  = Plotter()

    envs = [
        'dayan', 'dayan_2',
        'fourrooms', 'fourrooms_2',
        'gridroom', 'gridroom_2',
        'gridmaze', 'gridmaze_2'
    ]

    env_labels = [
        "Grid Task",# "Modified Grid Task (Reward)",
        "Four Rooms",# "Four Rooms (Reward)",
        "Grid Room", #"Modified Grid Room (Reward)",
        "Grid Maze",# "Modified Grid Maze (Reward)"
    ]

    modes = [
        "none",
        "SR_wang",
        "SR_potential",
        "DR_potential"
    ]

    labels = [
        "ns",
        "SR-pri",
        "SR-pot",
        "DR-pot"
    ]

    best_setting = {
    "dayan": {
        "none": [0., 0.3],
        "SR_wang": [0.25, 0.3],
        "SR_potential": [0.75, 1.0],
        "DR_potential": [0.75, 1.0]
    },
    "dayan_2": {
        "none": [0., 0.3],
        "SR_wang": [0.25, 0.3],
        "SR_potential": [0.5, 0.3],
        "DR_potential": [0.75, 1.0]
    },
    "fourrooms": {
        "none": [0., 0.3],
        "SR_wang": [0.25, 0.3],
        "SR_potential": [0.5, 0.3],
        "DR_potential": [0.75, 1.0]
    },
    "fourrooms_2": {
        "none": [0., 0.3],
        "SR_wang": [0.75, 1.0],
        "SR_potential": [0.25, 1.0],
        "DR_potential": [0.75, 1.0]
    },
    "gridroom": {
        "none": [0., 0.3],
        "SR_wang": [0.25, 0.3],
        "SR_potential": [0.25, 1.0],
        "DR_potential": [0.5, 1.0]
    },
    "gridroom_2": {
        "none": [0., 1.0],
        "SR_wang": [0.25, 1.0],
        "SR_potential": [0.25, 1.0],
        "DR_potential": [0.5, 1.0]
    },
    "gridmaze": {
        "none": [0., 1.0],
        "SR_wang": [0.25, 0.3],
        "SR_potential": [0.25, 1.0],
        "DR_potential": [0.5, 1.0]
    },
    "gridmaze_2": {
        "none": [0., 1.0],
        "SR_wang": [0.5, 1.0],
        "SR_potential": [0.25, 1.0,],
        "DR_potential": [0.5, 1.0]
    }
    }

    env_specific_axis_setting = [
        # [xlim low, xlim high, ylim low, ylim high]
        [0, 10000, -100, 5],
        [0, 10000, -400, 5],
        [0, 10000, -210, 5],
        [0, 10000, -210, 5],
        [-300, 60000, -1020, 5],
        [0, 60000, -1020, 5],      # -1000
        [-700, 100000, -1020, 5],
        [0, 100000, -1020, 5],
    ]

    plt.rcParams.update({
    'font.size': 12  # set your preferred default size here
    })

    fig, axs = plt.subplots(1, 4, figsize=(13, 2.8))
    axs = axs.T.flatten()

    plot_reward_env = False

    # plot only environments with low-reward regions
    if plot_reward_env:
        idx = ["2" in e for e in envs]
    else:
        idx = ["2" not in e for e in envs]

    envs = [x for x, m in zip(envs, idx) if m]
    env_specific_axis_setting = [x for x, m in zip(env_specific_axis_setting, idx) if m]


    for env, env_label, axis_setting, ax in zip(envs, env_labels, env_specific_axis_setting, axs):
        print(env)
        plotter.reset()
        for mode, label in zip(modes, labels):

            r_aux_w, step_size = best_setting[env][mode]
            timesteps, performance = read_data(env, mode, r_aux_w, step_size)
            # if mode == "SR_potential":
            # plotter.plot_data(ax, timesteps, performance, plot_conf_int=False, plot_all_seeds=True)
            plotter.plot_data(ax, timesteps, performance, plot_conf_int=True, plot_all_seeds=False)

            
            if env == "dayan":
                if mode == "none":
                    plotter.draw_text(ax, 6000, -80, label, size=12)
                elif mode == "SR_wang":
                    plotter.draw_text(ax, 1500, -60, label, size=12)
                elif mode == "SR_potential":
                    plotter.draw_text(ax, 500, -20, label, size=12)
                elif mode == "DR_potential":
                    plotter.draw_text(ax, 4500, -5, label, size=12)

            if env =="dayan_2":
                if mode == "none":
                    plotter.draw_text(ax, 5800, -65, label, size=12)
                elif mode == "SR_wang":
                    plotter.draw_text(ax, 6000, -360, label, size=12)
                elif mode == "SR_potential":
                    plotter.draw_text(ax, 2500, -60, label, size=12)
                elif mode == "DR_potential":
                        plotter.draw_text(ax, 7200, -4, label, size=12)

            plotter.index += 1

        xlabel = None
        ylabel = None

        if "dayan" in env:
            ylabel = "Return"

        if "2" in env:
            xlabel = "Thousand Steps"

        
        plotter.finalize_plot(ax, xlabel=xlabel, ylabel=ylabel, title=env_label, axis_setting=axis_setting)


        # x tick
        total_timesteps = axis_setting[1]
        interval = 4
        tick_positions = range(0, total_timesteps + 1, total_timesteps // 4)
        tick_labels = [x / 1000 for x in tick_positions]
        tick_labels = [int(x) if np.round(x) == x else x for x in tick_labels]
        ax.set_xticks(tick_positions, tick_labels)

        # y ticks
        min_return = axis_setting[2]
        min_return = int(np.ceil(min_return / 100) * 100)
        interval = 2
        tick_positions = range(0, min_return - 1, min_return // interval)
        ax.set_yticks(tick_positions)

    if plot_reward_env:
        fig_num = 4
    else:
        fig_num = 5
    plt.savefig(f"minigrid_basics/plots/Figure_{fig_num}.png", dpi=300, bbox_inches='tight')
    # plt.show()

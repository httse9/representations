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

def compute_p_r_stat(env_name, representation):
    """
    For environment and representation (DR/SR) pair,
    read the data for all hyperparameters.
    Throw away hyperparameter settings where eigendecomposition is unstable.
    Compute the AUC of state-visitation and average reward.
    """

    ps = {}
    rs = {}
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        seed_fail, r, p = read_data(env_name, representation, *hyper)

        hyper_strings = [str(v) for v in list(hyper)]
        hypername = '-'.join(hyper_strings)
        print(hypername)
        
        ps[hypername] = p
        rs[hypername] = r

    return ps, rs

## hyperparameters
p_option = [0.01, 0.05, 0.1]
dataset_size = [100, ]
learn_rep_iter = [1, 10, 100]      
rep_lr = [0.01, 0.03, 0.1]
num_options = [1, 8, 1000]

if __name__ == "__main__":

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2",]

    env_labels = [
        "Modified Grid Task",
        "Four Rooms",
        "Modified Grid Room",
        "Modified Grid Maze", 
    ]

    ### read data
    p_dict = {}
    r_dict = {}
    representation = ["SR", "DR"]

    for env_name in envs:
        p_dict[env_name] = {}
        r_dict[env_name] = {}

    for env_name in envs:
        # print(env_name)
        for rep in representation:
            # print(f"  {rep}")
            path = join(rod_directory, env_name, rep)

            try:    # try to read processed data if exists
                with open(join(path, "p.pkl"), "rb") as f:
                    p_dict[env_name][rep] = pickle.load(f)

                with open(join(path, "r.pkl"), "rb") as f:
                    r_dict[env_name][rep] = pickle.load(f)

            except: # process data and save

                ps, rs = compute_p_r_stat(env_name, rep)
            
                with open(join(path, "p.pkl"), "wb") as f:
                    pickle.dump(ps, f)

                with open(join(path, "r.pkl"), "wb") as f:
                    pickle.dump(rs, f)

                p_dict[env_name][rep] = ps
                r_dict[env_name][rep] = rs

    # read random walk data
    for env_name in envs:
        path = join(rod_directory, env_name)
        
        rw_r = []
        rw_p = []
        for s in range(1, 11):
            with open(join(path, f"random_walk_{s}.pkl"), "rb") as f:
                data = pickle.load(f)
                rw_r.append(data['rewards'])
                rw_p.append(data['visit_percentage'])

        p_dict[env_name]["rw"] = np.array(rw_p)
        r_dict[env_name]["rw"] = np.array(rw_r)

    plotter = Plotter()

    ### Figure 1
    ### Scatter of all low-reward envs
    keys = []
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        hyper_strings = [str(v) for v in list(hyper)]
        hypername = '-'.join(hyper_strings)
        keys.append(hypername)


    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    reward_envs = [e for e in envs if "2" in e]

    for env_name, env_label, ax in zip(reward_envs, env_labels, axs):
        for j, rep in enumerate(representation):
            plotter.index = j

            ps = p_dict[env_name][rep]
            rs = r_dict[env_name][rep]

            ps = [ps[key] for key in keys]
            rs = [rs[key] for key in keys]

            p_auc = [p.mean() for p in ps]
            r_avg = [r.mean() for r in rs]

            # plot data
            c = Colors.colors[j]
            ax.scatter(p_auc, r_avg, color=c, marker="o")

            ax.axvline(np.max(p_auc), color=c, linestyle='--', alpha=0.2)


            if "dayan" in env_name:
                if rep == "SR":
                    plotter.draw_text(ax, 0.93, -3.1, rep)
                elif rep == "DR":
                    plotter.draw_text(ax, 0.9, -2.5, rep)

        # random walk data
        rw_p = p_dict[env_name]["rw"].mean()
        rw_r = r_dict[env_name]["rw"].mean()
        ax.scatter(rw_p, rw_r, color=Colors.colors[2], marker="o")
        plotter.index = 2
        if "dayan" in env_name:
            plotter.draw_text(ax, 0.925, -3.85, "RW")

        if "dayan" in env_name:
            y_label = "Average Reward Per Timestep"
        else:
            y_label = None
        plotter.finalize_plot(ax, title=env_label, xlabel="State Visit Percentage AUC", ylabel=y_label)
        plt.tight_layout()

       
    plt.savefig(f"minigrid_basics/plots/rod_Figure_1.png", dpi=300)

    plt.close()

    ### Figure 2
    ### Scatter of all low-reward envs
    ### Focus on last 10% of state visit percentage
    keys = []
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        hyper_strings = [str(v) for v in list(hyper)]
        hypername = '-'.join(hyper_strings)
        keys.append(hypername)

    


    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    reward_envs = [e for e in envs if "2" in e]

    for env_name, env_label, ax in zip(reward_envs, env_labels, axs):
        for j, rep in enumerate(representation):
            plotter.index = j

            ps = p_dict[env_name][rep]
            rs = r_dict[env_name][rep]

            ps = [ps[key] for key in keys]
            rs = [rs[key] for key in keys]

            length = ps[0].shape[1] // 10

            p_auc = [p.mean(0)[-length:].mean() for p in ps]
            r_avg = [r.mean() for r in rs]

            # plot data
            c = Colors.colors[j]
            ax.scatter(p_auc, r_avg, color=c, marker="o")

            ax.axvline(np.max(p_auc), color=c, linestyle='--', alpha=0.2)

            if "dayan" in env_name:
                if rep == "SR":
                    plotter.draw_text(ax, 0.99, -3.1, rep)
                elif rep == "DR":
                    plotter.draw_text(ax, 0.975, -2.5, rep)

        # random walk data
        rw_p = p_dict[env_name]["rw"].mean(0)[-length:].mean()
        rw_r = r_dict[env_name]["rw"].mean()
        ax.scatter(rw_p, rw_r, color=Colors.colors[2], marker="o")
        plotter.index = 2
        if "dayan" in env_name:
            plotter.draw_text(ax, 0.99, -3.9, "RW")

        if "dayan" in env_name:
            y_label = "Average Reward Per Timestep"
        else:
            y_label = None
        plotter.finalize_plot(ax, title=env_label, xlabel="State Visit Percentage AUC", ylabel=y_label)


       
    plt.savefig(f"minigrid_basics/plots/rod_Figure_2.png", dpi=300)

    
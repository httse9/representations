import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter, Colors
from itertools import product

plt.rcParams.update({
'font.size': 12  # set your preferred default size here
})

plotter = Plotter()
fig, axs = plt.subplots(2, 4, figsize=(11, 5))
axs = axs.flatten()


### first row
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

envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2",]

env_labels = [
    "Grid Task",
    "Four Rooms",
    "Grid Room",
    "Grid Maze", 
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

### Figure 1
### Scatter of all low-reward envs
keys = []
for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
    hyper_strings = [str(v) for v in list(hyper)]
    hypername = '-'.join(hyper_strings)
    keys.append(hypername)


reward_envs = [e for e in envs if "2" in e]

for env_name, env_label, ax in zip(reward_envs, env_labels, axs[:4]):
    best_p, best_r = [[], []], [[], []]
    for j, rep in enumerate(representation):
        plotter.index = j

        ps = p_dict[env_name][rep]
        rs = r_dict[env_name][rep]

        ps = [ps[key] for key in keys]
        rs = [rs[key] for key in keys]

        p_auc = np.array([p.mean() for p in ps])
        r_avg = np.array([r.mean() for r in rs])

        # plot data
        c = Colors.colors[j]
        ax.scatter(p_auc, r_avg, color=c, marker="o", alpha=0.1)

        # record best points
        k = 1
        top_idx = np.argpartition(p_auc, -k)[-k:]
        for idx in top_idx:
            print(env_name, rep, keys[idx])
            best_p[j].append(p_auc[idx])
            best_r[j].append(r_avg[idx])

        if "dayan" in env_name:
            if rep == "SR":
                plotter.draw_text(ax, 0.92, -3.1, "CEO")
            elif rep == "DR":
                plotter.draw_text(ax, 0.88, -2.5, "RACE")

    for j, (bp, br) in enumerate(zip(best_p, best_r)):
        c = Colors.colors[j]
        ax.axvline(np.max(bp), color=c, linestyle='--', alpha=0.2)
        ax.scatter(bp, br, color=c, marker="o", alpha=1.)

    # random walk data
    rw_p = p_dict[env_name]["rw"].mean()
    rw_r = r_dict[env_name]["rw"].mean()
    ax.scatter(rw_p, rw_r, color=Colors.colors[2], marker="o")
    plotter.index = 2
    if "dayan" in env_name:
        plotter.draw_text(ax, 0.925, -3.85, "RW")

    if "dayan" in env_name:
        y_label = "Avg. Reward"
    else:
        y_label = None
    plotter.finalize_plot(ax, title=env_label, xlabel="Avg. State Visitation %", ylabel=y_label)




### second row of plot
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

        if s == 1:
            print(path, filename)

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
    [0.1, 100, 100, 0.01, 1],
    [0.05, 100, 100, 0.01, 1],
    [0.05, 100, 100, 0.01, 1]
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
    [0.05, 100, 1, 0.03, 1],
    [0.1, 100, 1, 0.03, 1]
]

qlearning_best_hyperparameters = [
    [-10, 0.2, 0.01],
    [-10, 0.15, 0.01],
    [-10, 0.2, 1.0],
    [-10, 0.15, 1.0],
    [-10, 0.2, 0.3],
    [-10, 0.2, 0.3],
    [-10, 0.2, 0.3],
    [0, 0.2, 1.0],
    [-10, 0.2, 0.3],
    [-10, 0.2, 0.3]
]

envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2", "gridroom_25", "gridmaze_29"]

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

idx = [True if e in ["gridroom_2", "gridmaze_2", "gridroom_25", "gridmaze_29"] else False for e in envs ]
print(idx)
envs = [x for x, m in zip(envs, idx) if m]
print(envs)
env_labels = ["Grid Room", "Grid Maze", "Grid Room (L)", "Grid Maze (L)"]
SR_best_hyperparameters = [x for x, m in zip(SR_best_hyperparameters, idx) if m]
DR_best_hyperparameters = [x for x, m in zip(DR_best_hyperparameters, idx) if m]
qlearning_best_hyperparameters = [x for x, m in zip(qlearning_best_hyperparameters, idx) if m]

axis_setting = [
    [None, None, -101, -70],
    [None, None, -101, -70],
    [None, None, -201, -75],
    [None, None, -201, -140],
]


for env_name, env_label, ax, SR_hyper, DR_hyper, q_hyper, axis_s in zip(envs, env_labels, axs[4:], SR_best_hyperparameters, DR_best_hyperparameters, qlearning_best_hyperparameters, axis_setting):

    print(env_name)
    _, q_sr = read_data(env_name, "SR_nt", *SR_hyper)
    _, q_dr = read_data(env_name, "DR", *DR_hyper)
    q_q = read_q_data(env_name, *q_hyper)

    print(q_sr.shape, q_dr.shape, q_q.shape)

    # print(q_sr[:, -1].argmin())

    max_performance = max(q_sr.mean(0).max(), q_dr.mean(0).max(), q_q.mean(0).max())

    
    x = np.array(range(q_sr.shape[1])) * 1.0

    plotter.index = 2
    plotter.plot_data(ax, x, q_q, plot_conf_int=True)

    plotter.index = 0
    plotter.plot_data(ax, x, q_sr, plot_conf_int=True, plot_all_seeds=False)

    plotter.index = 1
    plotter.plot_data(ax, x, q_dr, plot_conf_int=True, plot_all_seeds=False)


    if env_name == "gridroom_2":
        y_label = "Return"
    else:
        y_label = None

    if env_name == "gridroom_2":
        plotter.index = 0
        plotter.draw_text(ax, 85, -83, "CEO+Q")
        plotter.index = 1
        plotter.draw_text(ax, 50, -75, "RACE+Q")
        plotter.index = 2
        plotter.draw_text(ax, 100, -98, "QL")

    if "2" in env_name:
        x_label = "Number of Iterations"
    else:
        x_label = None
        

    plotter.finalize_plot(ax, title=env_label, xlabel=x_label, ylabel=y_label, axis_setting=axis_s)

plt.savefig(f"minigrid_basics/plots/rod_Figure_7.png", dpi=300)
plt.close()


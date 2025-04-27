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

# check how many seeds are present
def check_seed(env_name, representation, p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed=10):
    path = join(rod_directory, env_name, representation)

    num_successful_seeds = 0

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)
        num_successful_seeds += int(os.path.isfile(join(path, filename)))
    
    seed_missing = num_successful_seeds < seed  # whether some seeds are missing

    return seed_missing, num_successful_seeds

## hyperparameters
p_option = [0.01, 0.05, 0.1]
dataset_size = [100, 100000]
learn_rep_iter = [1, 10, 100]      
rep_lr = [0.01, 0.03, 0.1]
num_options = [1, 8, 1000]


def check_seed_env(env, rep):
    """
    Check for which hyperparameter settings some seeds are missing
    """
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        seed_missing, ss = check_seed(env, rep, *hyper)

        if seed_missing:
            print("  ", ss, hyper, build_command(env, rep, *hyper))

def build_command(env, rep, p_option, dataset_size, learn_rep_iter, rep_lr, num_options):

    return f"sbatch --array=1-10 rod.sh {env} {rep} {p_option} {dataset_size} {learn_rep_iter} {rep_lr} {num_options}"

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



if __name__ == "__main__":
    """
    TODO:
    1. Figure out for which hyperparameter setting, seeds are missing
        a) simplest: print out those
        b) more advanced: 
    """

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2",]

    ### check seeds
    # for env in ["gridroom_2"]:
    #     print(">>>>>>>>>", env, flush=True)
    #     check_seed_env(env, "DR")


    ### process data
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


    ### plot
    plotter = Plotter()


    # generate plots
    dataset_size = [100, ]
    keys = []
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        hyper_strings = [str(v) for v in list(hyper)]
        hypername = '-'.join(hyper_strings)
        keys.append(hypername)


    for env_name in envs:
        print(env_name)
        fig, ax = plt.subplots()
        for j, rep in enumerate(representation):
            print(f"  {rep}: ", end="")
            ps = p_dict[env_name][rep]
            rs = r_dict[env_name][rep]

            # keys = ps.keys()
            ps = [ps[key] for key in keys]
            rs = [rs[key] for key in keys]

            p_auc = []
            r_avg = []

            p_auc_sf = []   # seed fail
            r_avg_sf = []

            for p, r in zip(ps, rs):
                # if p.shape[0] < 10:       # some seeds failed, ignore
                #     continue
                
                if p.shape[0] == 10:
                    p_auc.append(p.mean(0).mean())    # state visitation AUC
                    # p_auc.append(p.mean(0)[-1])     # final state visitation percentage
                    r_avg.append(r.mean(0).mean())

                else:
                    # failed seeds
                    p_auc_sf.append(p.mean(0).mean())
                    r_avg_sf.append(r.mean(0).mean())

            # plot data
            c = Colors.colors[j]
            plt.scatter(p_auc, r_avg, color=c, label=f"{rep}", marker="o")

            # plot best hyperparameter setting
            # find out best visit
            idx = np.argmax(p_auc)
            # print(p_auc[idx], r_avg[idx])
            plt.scatter([p_auc[idx]], [r_avg[idx]], color="k", marker="*", s=12)
            # assert len(p_auc) == len(keys)
            print(list(keys)[idx])        # assume no seed fail

            # To plot failed seeds, uncomment:
            # plt.scatter(p_auc_sf, r_avg_sf, color=c, label=f"{rep} (fail)", marker="x")

            plt.axvline(np.max(p_auc), color=c, linestyle='--', alpha=0.2)

        plotter.finalize_plot(ax, title=env_name, xlabel="State Visit AUC", ylabel="Average Reward Per Timestep")
        plt.legend()

       
        plt.savefig(f"{rod_directory}/{env_name}.png")

        plt.clf()

    
    dataset_size = [100, 100000]
    hyperparameter_list = [p_option, dataset_size, learn_rep_iter, rep_lr, num_options]
    hyperparameter_names = ["p_option", "dataset_size", "learn_rep_iter", "rep_lr", "num_options"]
    # see impact each hyperparameter 
    for env_name in envs:
        print(env_name)

        for i, rep in enumerate(representation):

            ps = p_dict[env_name][rep]
            rs = r_dict[env_name][rep]

            # enumerate over hyperparameters
            for j, (hyperparameter, hyperparameter_name) in enumerate(zip(hyperparameter_list, hyperparameter_names)):

                fix, ax = plt.subplots()

                # for the chosen hyperparameter, iterate over values
                for h, c in zip(hyperparameter, Colors.colors):

                    hyper_list = [p_option, dataset_size, learn_rep_iter, rep_lr, num_options]
                    hyper_list[j] = [h]

                    # generate keys for each hyperparameter value
                    hypernames = []
                    for hyper in product(*hyper_list):
                        hyper_strings = [str(v) for v in list(hyper)]
                        hypername = '-'.join(hyper_strings)
                        hypernames.append(hypername)

                    p_hyper = [ps[hypername] for hypername in hypernames]
                    r_hyper = [rs[hypername] for hypername in hypernames]

                    p_auc = [p.mean() for p in p_hyper]
                    r_avg = [r.mean() for r in r_hyper]

                    ax.scatter(p_auc, r_avg, color=c, label=h)

                plt.legend()
                plotter.finalize_plot(ax, title=f"{env_name}_{rep}_{hyperparameter_name}", xlabel="State Visit AUC", ylabel="Average Reward Per Timestep")

                plt.savefig(f"{rod_directory}/{env_name}_{rep}_{hyperparameter_name}.png")
                plt.clf()

                






    

            

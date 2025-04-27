import numpy as np
import pickle
from os.path import join
from itertools import product
from tqdm import tqdm

path = join("minigrid_basics", "experiments", "rod_q")

# Q baseline hyperparameters
initialization = [-1000, -100, -10, 0]
epsilon = [0.01, 0.05, 0.1, 0.15, 0.2]
stepsize = [0.01, 0.03, 0.1, 0.3, 1.0]

def construct_filename(init, eps, stepsize, seed):
    values = [init, eps, stepsize]
    values = [str(v) for v in values]
    filename = '-'.join(values) + f"-{seed}.pkl"
    return filename

def read_data(env, init, eps, stepsize):

    fpath = join(path, env, "qlearning")
    q_performance = []
    for s in range(1, 11):

        filename = construct_filename(init, eps, stepsize, s)
        with open(join(fpath, filename), "rb") as f:
            data = pickle.load(f)
        q_performance.append(data['Q_performance'])

    return np.array(q_performance)
        

if __name__ == "__main__":

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2",]

    # enumerate over environments
    for env in envs:
        print(f"Environments: {env}")

        # keep track of best performance & hyper
        best_hyper = None
        best_return_final = -1000000000
        best_return_mean = -100000000

        # enumerate over hyperparameters
        for init, eps, steps in tqdm(product(initialization, epsilon, stepsize)):
            
            # read data
            q_performance = read_data(env, init, eps, steps)

            # mean over seeds
            q_performance = q_performance.mean(0)

            return_mean = q_performance.mean()
            return_final = q_performance[-1]

            # if final performance better or final same but learn faster
            # record hyperparameter settings
            if return_final > best_return_final or (return_final == best_return_final and return_mean > best_return_mean):
                best_hyper = [init, eps, steps]
                best_return_final = return_final
                best_return_mean = return_mean

        print("  Best Hyperparameters:", best_hyper)
        print("  Best Return Final:", best_return_final)
        print("  Best Return Mean:", best_return_mean)


# Best hyperparameters
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
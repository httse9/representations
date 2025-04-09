import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from examples.plotter import Plotter, Colors
from itertools import product

# build the file name given the hyperparameters
def construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed):
    values = [p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options]
    values = [str(v) for v in values]
    filename = '-'.join(values) + f"-{seed}.pkl"
    return filename

# check how many seeds are present
def check_seed(env_name, representation, p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed=10):
    path = join("experiments", "rod", env_name, representation)


    successful_seeds = 0

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)
        # print(filename, os.path.isfile(join(path, filename)))
        # print(join(path, filename))

        successful_seeds += int(os.path.isfile(join(path, filename)))

    seed_missing = successful_seeds < 10
    # print(successful_seeds)

    return seed_missing, successful_seeds


## hyperparameters
p_option = [0.01, 0.05, 0.1]
dataset_size = [100, 100000]
learn_rep_iter = [1, 10, 100]
rep_lr = [0.01, 0.03, 0.1]
num_options = [1, 8, 1000]


# check if some seeds missing
for x in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):

    seed_missing, ss = check_seed("gridroom", "DR", *x)

    if seed_missing:
        print(x, ss)

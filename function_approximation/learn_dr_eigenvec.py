"""
Learn eigenvector of DR directly
"""

import numpy as np
import jax.numpy as jnp
import random


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms", type=str, help="Specify environment.")
    parser.add_argument("--obs_type", default="onehot", type=str, help="Type of environment observation")
    parser.add_argument("--n_epochs", type=int, default=10000)

    # hyperparams
    parser.add_argument("--step_size_start", default=1e-4, type=float, help="Starting step size")
    parser.add_argument("--step_size_end", default=3e-5, type=float, help="Ending step size")
    parser.add_argument("--grad_norm_clip", default=0.5, type=float, help="Gradient norm clipping")

    parser.add_argument("--seed", default=0, type=int, help="Initial random seed/key")

    args = parser.parse_args()
    set_random_seed(args.seed)




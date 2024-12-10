#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-machado

cd ~
module load python/3.10
source REP_ENV2/bin/activate

cd representations


python -m minigrid_basics.examples.reward_shaping --representation baseline --seed 2

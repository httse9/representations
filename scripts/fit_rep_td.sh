#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-machado
#SBATCH --output=slurm_out/slurm-%x.%j.out

cd ~
module load python/3.10
source REP_ENV2/bin/activate

cd representations

python -m minigrid_basics.examples.fit_rep_td --env $1 --n_episodes $2 --lr $3 --seed $SLURM_ARRAY_TASK_ID 

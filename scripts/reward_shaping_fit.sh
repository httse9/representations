#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-machado
#SBATCH --output=slurm_out/slurm-%x.%j.out

cd ~
module load python/3.10
source REP_ENV2/bin/activate

cd representations

python -m minigrid_basics.examples.reward_shaping_fit --env $1 --representation $2 --n_episodes $3 --r_shaped_weight $4 --lr $5 --seed $SLURM_ARRAY_TASK_ID 

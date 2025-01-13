#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-machado
#SBATCH --output=slurm_out/slurm-%x.%j.out


cd ~
module load python/3.10
source REP_ENV2/bin/activate

cd representations



# run exp
python -m minigrid_basics.examples.reward_shaping --representation $2 --i_eigen $5 --r_shaped_weight $3 --lr $4 --seed $SLURM_ARRAY_TASK_ID --env $1



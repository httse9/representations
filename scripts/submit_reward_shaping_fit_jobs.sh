#!/bin/bash

for EXP_ENV in maxent_empty # maxent_empty maxent_empty_2 maxent_fourrooms maxent_fourrooms_2 
do
  for REP in MER SR 
  do
    for N_EPISODES in 100 500 1000
    do
      for R_SHAPED_WEIGHT in 0.25 0.5 0.75 1.0
      do
        for LR in 0.1 0.3 1.0 #3.0
        do
          sbatch --array=1-20 reward_shaping_fit.sh $EXP_ENV $REP $N_EPISODES $R_SHAPED_WEIGHT $LR 
        done
      done
    done
  done
done

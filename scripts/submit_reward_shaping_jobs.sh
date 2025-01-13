#!/bin/bash

for EXP_ENV in maxent_empty maxent_empty_2 maxent_fourrooms maxent_fourrooms_2 
do
  for REP in SR MER 
  do
    for R_SHAPED_WEIGHT in 0.25 0.5 0.75 1.0
    do
      for LR in 0.03 0.1 0.3 1.0 #3.0
      do
        sbatch --array=1-20 reward_shaping.sh $EXP_ENV $REP $R_SHAPED_WEIGHT $LR 0 
      done
    done
  done
done

for EXP_ENV in maxent_empty maxent_empty_2 maxent_fourrooms maxent_fourrooms_2
do
  for REP in baseline
  do
    for R_SHAPED_WEIGHT in 0.0
    do
      for LR in 0.03 0.1 0.3 1.0 #3.0
      do
        sbatch --array=1-20 reward_shaping.sh $EXP_ENV $REP $R_SHAPED_WEIGHT $LR 0
      done
    done
  done
done

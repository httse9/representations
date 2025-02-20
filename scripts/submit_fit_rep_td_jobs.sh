#!/bin/bash

for EXP_ENV in maxent_fourrooms_2 #maxent_empty_2 maxent_empty maxent_fourrooms
do
  for N_EPI in 100 #500 1000 5000 
  do
    for LR in  0.01 #0.03 0.1 0.3 1.0 3.0
    do
      sbatch --array=1-20 fit_rep_td.sh $EXP_ENV $N_EPI $LR 
    done
  done
done

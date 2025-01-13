#!/bin/bash

for EXP_ENV in maxent_fourrooms maxent_fourrooms_2
do
  for REP in SR DR MER 
  do
    for LR in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1.0
    do
      sbatch --array=1-20 learn_rep_td.sh $EXP_ENV $REP $LR 
    done
  done
done

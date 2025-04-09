#!/bin/bash

losses=('NLL' 'MMDf' 'MMDa', wasserstein)
priors=('True' 'False')

for loss in "${losses[@]}"; do
    for prior in "${priors[@]}"; do
        python train_figs.py "$loss" "$prior" "ik_robotics"
    done
done
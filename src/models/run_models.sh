#!/bin/bash

models=("LGBM", "LSTM", "CoxPHM")
processes=("tune", "train", "tune")

for model in ${models[@]}; do
    for process in ${processes[@]}; do
        echo $model, $process
        python3 main.py --model $model --process $process
    done
done

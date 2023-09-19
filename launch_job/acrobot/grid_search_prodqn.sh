#!/bin/bash


for TARGET_UPDATES in 60 90
do
    for N_WEIGHTS in 1 10
    do  
        for WEIGHTS_UPDATES in 2000 5000
        do 
            EXPERIMENT_NAME=pbo2_2_2_2_lr5_t$TARGET_UPDATES\_w$N_WEIGHTS\_u$WEIGHTS_UPDATES

            PARAMS=$(jq '.prodqn_n_training_steps_per_target_update = $TARGET_UPDATES' --argjson TARGET_UPDATES $TARGET_UPDATES experiments/acrobot/parameters.json)
            PARAMS=$(jq '.prodqn_n_current_weights = $N_WEIGHTS' --argjson N_WEIGHTS $N_WEIGHTS <<<"$PARAMS")
            PARAMS=$(jq '.prodqn_n_training_steps_per_current_weight_update = $WEIGHTS_UPDATES' --argjson WEIGHTS_UPDATES $WEIGHTS_UPDATES <<<"$PARAMS")
            echo $PARAMS > experiments/acrobot/parameters.json

            launch_job/acrobot/launch_prodqn.sh -e $EXPERIMENT_NAME -fs 1 -ls 1 -b 10
        done
    done
done
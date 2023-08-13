#!/bin/bash


for TARGET_UPDATES in 30 100
do
    for N_WEIGHTS in 1 20
    do  
        for WEIGHTS_UPDATES in 3000 5000
        do 
            EXPERIMENT_NAME=pbo2_2_2_2_lr5_t$TARGET_UPDATES\_w$N_WEIGHTS\_u$WEIGHTS_UPDATES

            PARAMS=$(jq '.prodqn_n_training_steps_per_target_update = $TARGET_UPDATES' --argjson TARGET_UPDATES $TARGET_UPDATES experiments/lunar_lander/parameters.json)
            PARAMS=$(jq '.prodqn_n_current_weights = $N_WEIGHTS' --argjson N_WEIGHTS $N_WEIGHTS <<<"$PARAMS")
            PARAMS=$(jq '.prodqn_n_training_steps_per_current_weight_update = $WEIGHTS_UPDATES' --argjson WEIGHTS_UPDATES $WEIGHTS_UPDATES <<<"$PARAMS")
            echo $PARAMS > experiments/lunar_lander/parameters.json

            launch_job/lunar_lander/launch_prodqn.sh -e $EXPERIMENT_NAME -fs 1 -ls 1 -b 5
        done
    done
done
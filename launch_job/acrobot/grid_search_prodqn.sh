#!/bin/bash


for TARGET_UPDATES in 30 100
do
    for WEIGHTS_UPDATES in 4000 6000
    do 
        EXPERIMENT_NAME=pbo2_2_2_2_lr5_t$TARGET_UPDATES\_w15_u$WEIGHTS_UPDATES

        PARAMS=$(jq '.prodqn_n_training_steps_per_target_update = $TARGET_UPDATES' --argjson TARGET_UPDATES $TARGET_UPDATES experiments/acrobot/parameters.json)
        PARAMS=$(jq '.prodqn_n_training_steps_per_current_weight_update = $WEIGHTS_UPDATES' --argjson WEIGHTS_UPDATES $WEIGHTS_UPDATES <<<"$PARAMS")
        echo $PARAMS > experiments/acrobot/parameters.json

        launch_job/acrobot/launch_prodqn.sh -e $EXPERIMENT_NAME -fs 1 -ls 1 -b 5
    done
done
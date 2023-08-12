#!/bin/bash


for REPLAY_BUFFER_SIZE in 20000 100000
do
    for DURATION_EPS in 50000 300000
    do 
        for BATCH_SIZE in 64 128 512
        do
            for TARGET_UPDATES in 500 1000 6000
            do 
                EXPERIMENT_NAME=r$REPLAY_BUFFER_SIZE\_d$DURATION_EPS\_b$BATCH_SIZE\_t$TARGET_UPDATES

                PARAMS=$(jq '.replay_buffer_size = $REPLAY_BUFFER_SIZE' --argjson REPLAY_BUFFER_SIZE $REPLAY_BUFFER_SIZE experiments/acrobot/parameters.json)
                PARAMS=$(jq '.duration_eps = $DURATION_EPS' --argjson DURATION_EPS $DURATION_EPS <<<"$PARAMS")
                PARAMS=$(jq '.batch_size = $BATCH_SIZE' --argjson BATCH_SIZE $BATCH_SIZE <<<"$PARAMS")
                PARAMS=$(jq '.dqn_n_training_steps_per_target_update = $TARGET_UPDATES' --argjson TARGET_UPDATES $TARGET_UPDATES <<<"$PARAMS")
                echo $PARAMS > experiments/acrobot/parameters.json

                launch_job/acrobot/launch_dqn.sh -e $EXPERIMENT_NAME -fs 1 -ls 1
            done
        done
    done
done
#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/bicycle_online/figures/$EXPERIMENT_NAME
[ -f experiments/bicycle_online/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/bicycle_online/parameters.json experiments/bicycle_online/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/DQN
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_deep


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $DQN = true ]]
    then
        # DQN
        echo "launch train dqn"
        bicycle_online_dqn -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        bicycle_online_dqn_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 


    if [[ $PBO_linear = true ]]
    then
        # PBO linear
        echo "launch train pbo linear"
        bicycle_online_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

        echo "launch evaluate pbo linear"
        bicycle_online_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear
    fi


    if [[ $PBO_deep = true ]]
    then
        # PBO deep
        echo "launch train pbo deep"
        bicycle_online_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

        echo "launch evaluate pbo deep"
        bicycle_online_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep
    fi
done
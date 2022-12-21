#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lunar_lander/figures/$EXPERIMENT_NAME
[ -f experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lunar_lander/parameters.json experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/DQN
# [ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_linear
# [ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_deep
# [ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/IDQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/IDQN

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $DQN = true ]]
    then
        # DQN
        echo "launch train dqn"
        lunar_lander_dqn -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        # lunar_lander_dqn_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 


    # if [[ $PBO_linear = true ]]
    # then
    #     # PBO linear
    #     echo "launch train pbo linear"
    #     lunar_lander_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

    #     echo "launch evaluate pbo linear"
    #     lunar_lander_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear
    # fi


    # if [[ $PBO_deep = true ]]
    # then
    #     # PBO deep
    #     echo "launch train pbo deep"
    #     lunar_lander_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

    #     echo "launch evaluate pbo deep"
    #     lunar_lander_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep
    # fi


    # if [[ $IDQN = true ]]
    # then
    #     # IDQN
    #     echo "launch train iDQN"
    #     lunar_lander_iDQN -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

    #     echo "launch evaluate iDQN"
    #     lunar_lander_iDQN_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    # fi
done
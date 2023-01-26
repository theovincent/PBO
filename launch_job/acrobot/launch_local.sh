#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/acrobot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/acrobot/figures/$EXPERIMENT_NAME
[ -f experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/acrobot/parameters.json experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/DQN
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/IDQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/IDQN

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $DQN = true ]]
    then
        # DQN
        echo "launch train dqn"
        acrobot_dqn -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        echo "launch evaluate dqn"
        acrobot_dqn_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 


    if [[ $PBO_linear = true ]]
    then
        # PBO linear
        echo "launch train pbo linear"
        acrobot_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

        echo "launch evaluate pbo linear"
        acrobot_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear
    fi


    if [[ $PBO_deep = true ]]
    then
        # PBO deep
        echo "launch train pbo deep"
        acrobot_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

        echo "launch evaluate pbo deep"
        acrobot_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep
    fi

    if [[ $IDQN = true ]]
    then
        # IDQN
        echo "launch train idqn"
        acrobot_idqn -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        echo "launch evaluate idqn"
        acrobot_idqn_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 
done
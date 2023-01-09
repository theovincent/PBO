#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/chain_walk/figures/$EXPERIMENT_NAME
[ -f experiments/chain_walk/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/chain_walk/parameters.json experiments/chain_walk/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/LSPI ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/LSPI
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_max_linear ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_max_linear
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_optimal ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_optimal


# Collect data
echo "launch collect sample"
chain_walk_sample -e $EXPERIMENT_NAME

if [[ $LSPI = true ]]
then
    # LSPI
    echo "launch train and evaluate lspi"
    chain_walk_lspi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION
fi 

if [[ $PBO_optimal = true ]]
then
    # PBO optimal
    echo "launch evaluate pbo optimal"
    chain_walk_pbo_optimal_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION
fi 

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $FQI = true ]]
    then
        # FQI
        echo "launch train fqi"
        chain_walk_fqi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        chain_walk_fqi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 


    if [[ $PBO_linear = true ]]
    then
        # PBO linear
        echo "launch train pbo linear"
        chain_walk_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

        echo "launch evaluate pbo linear"
        chain_walk_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear
    fi

    if [[ $PBO_max_linear = true ]]
    then
        # PBO deep
        echo "launch train pbo max linear"
        chain_walk_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a max_linear

        echo "launch evaluate pbo max linear"
        chain_walk_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a max_linear
    fi

    if [[ $PBO_deep = true ]]
    then
        # PBO deep
        echo "launch train pbo deep"
        chain_walk_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

        echo "launch evaluate pbo deep"
        chain_walk_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep
    fi
done
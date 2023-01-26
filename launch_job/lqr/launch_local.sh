#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lqr/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lqr/figures/$EXPERIMENT_NAME
[ -f experiments/lqr/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lqr/parameters.json experiments/lqr/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_custom_linear ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_custom_linear
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_optimal ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_optimal


# Collect data
echo "launch collect sample"
lqr_sample -e $EXPERIMENT_NAME


if [[ $PBO_optimal = true ]]
then
    # PBO optimal
    echo "launch evaluate pbo optimal"
    lqr_pbo_optimal_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -v 6
fi 

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $FQI = true ]]
    then
        # FQI
        echo "launch train fqi"
        lqr_fqi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        lqr_fqi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 

    if [[ $PBO_linear = true ]]
    then
        # PBO linear
        echo "launch train pbo linear"
        lqr_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

        echo "launch evaluate pbo linear"
        lqr_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear -v 6
    fi

    if [[ $PBO_custom_linear = true ]]
    then
        # PBO deep
        echo "launch train pbo max linear"
        lqr_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a custom_linear

        echo "launch evaluate pbo max linear"
        lqr_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a custom_linear  -v 6
    fi

    if [[ $PBO_deep = true ]]
    then
        # PBO deep
        echo "launch train pbo deep"
        lqr_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

        echo "launch evaluate pbo deep"
        lqr_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep -v 6
    fi
done
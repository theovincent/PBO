#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/bicycle_offline/figures/$EXPERIMENT_NAME
[ -f experiments/bicycle_offline/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/bicycle_offline/parameters.json experiments/bicycle_offline/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/IFQI ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/IFQI


# Collect data
echo "launch collect sample"
bicycle_offline_sample -e $EXPERIMENT_NAME

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $FQI = true ]]
    then
        # FQI
        echo "launch train fqi"
        bicycle_offline_fqi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        bicycle_offline_fqi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 


    if [[ $PBO_linear = true ]]
    then
        # PBO linear
        echo "launch train pbo linear"
        bicycle_offline_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

        echo "launch evaluate pbo linear"
        bicycle_offline_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear
    fi


    if [[ $PBO_deep = true ]]
    then
        # PBO deep
        echo "launch train pbo deep"
        bicycle_offline_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

        echo "launch evaluate pbo deep"
        bicycle_offline_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep
    fi


    if [[ $IFQI = true ]]
    then
        # IFQI
        echo "launch train ifqi"
        bicycle_offline_ifqi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        echo "launch evaluate ifqi"
        bicycle_offline_ifqi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi
done
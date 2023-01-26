#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -f experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/IFQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/IFQI


# Collect data
echo "launch collect sample"
car_on_hill_sample -e $EXPERIMENT_NAME -c

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    if [[ $FQI = true ]]
    then
        # FQI
        echo "launch train fqi"
        car_on_hill_fqi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        car_on_hill_fqi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi 


    if [[ $PBO_linear = true ]]
    then
        # PBO linear
        echo "launch train pbo linear"
        car_on_hill_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear

        echo "launch evaluate pbo linear"
        car_on_hill_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a linear
    fi


    if [[ $PBO_deep = true ]]
    then
        # PBO deep
        echo "launch train pbo deep"
        car_on_hill_pbo -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep

        echo "launch evaluate pbo deep"
        car_on_hill_pbo_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed -a deep
    fi


    if [[ $IFQI = true ]]
    then
        # IFQI
        echo "launch train ifqi"
        car_on_hill_ifqi -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed

        echo "launch evaluate ifqi"
        car_on_hill_ifqi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -s $seed
    fi
done
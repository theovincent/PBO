#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/FQI
[ -f experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json

# Collect data
echo "launch collect sample"
car_on_hill_sample -e $EXPERIMENT_NAME

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    # FQI
    echo "launch train fqi"
    car_on_hill_fqi -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE -s $seed

    echo "launch evaluate fqi"
    car_on_hill_fqi_evaluate -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE -s $seed
done
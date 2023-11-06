#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/ProFQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/ProFQI
[ -f experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json

# Collect data
echo "launch collect sample"
car_on_hill_sample -e $EXPERIMENT_NAME

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    # ProFQI
    echo "launch train profqi"
    car_on_hill_profqi -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE -s $seed

    echo "launch evaluate profqi"
    car_on_hill_profqi_evaluate -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE -s $seed
done
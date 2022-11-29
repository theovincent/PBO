#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

if [[ $COUNT_SAMPLES = true ]]
then
    car_on_hill_sample -e $EXPERIMENT_NAME -c
else
    car_on_hill_sample -e $EXPERIMENT_NAME
fi
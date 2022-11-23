#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

car_on_hill_pbo_evaluate -e $EXPERIMENT_NAME -s $SEED -b $MAX_BELLMAN_ITERATION -a linear
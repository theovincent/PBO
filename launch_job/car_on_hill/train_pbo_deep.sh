#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_gpu/bin/activate 

car_on_hill_pbo -s $SEED -b $MAX_BELLMAN_ITERATION -a deep
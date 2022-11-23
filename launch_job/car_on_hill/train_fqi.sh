#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_gpu/bin/activate 

car_on_hill_fqi -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID -b $MAX_BELLMAN_ITERATION
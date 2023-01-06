#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

chain_walk_fqi_evaluate -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID -b $MAX_BELLMAN_ITERATION
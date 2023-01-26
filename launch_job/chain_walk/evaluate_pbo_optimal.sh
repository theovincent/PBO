#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

chain_walk_pbo_optimal_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -v 50
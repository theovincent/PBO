#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

chain_walk_lspi_evaluate -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION
#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

chain_walk_sample -e $EXPERIMENT_NAME 
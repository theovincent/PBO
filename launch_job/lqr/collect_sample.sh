#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

lqr_sample -e $EXPERIMENT_NAME 
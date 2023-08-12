#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env/bin/activate

acrobot_prodqn -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID -b $BELLMAN_ITERATIONS_SCOPE
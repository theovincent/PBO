#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env/bin/activate

atari_dqn -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID
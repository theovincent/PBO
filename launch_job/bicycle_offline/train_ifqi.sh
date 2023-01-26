#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi 

bicycle_offline_ifqi -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID -b $MAX_BELLMAN_ITERATION
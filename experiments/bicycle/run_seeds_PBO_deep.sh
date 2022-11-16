#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/bicycle/parameters.json -k max_bellman_iterations -v $N_BI

# PBO linear max linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    python3 experiments/bicycle/PBO_linear_max_linear.py
    echo PBO linear max linear: $counter out of $N_SEEDS runs
    ((counter++))
done

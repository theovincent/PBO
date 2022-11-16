#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/bicycle/parameters.json -k max_bellman_iterations -v $N_BI

# FQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/bicycle/FQI.ipynb
    echo FQI: $counter out of $N_SEEDS runs
    ((counter++))
done
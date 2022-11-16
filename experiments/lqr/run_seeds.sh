#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/lqr/parameters.json -k max_bellman_iterations -v $N_BI

# FQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/FQI.ipynb
    echo FQI: $counter out of $N_SEEDS runs
    ((counter++))
done

# LSPI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/LSPI.ipynb
    echo LSPI: $counter out of $N_SEEDS runs
    ((counter++))
done

# optimal
jupyter nbconvert --to notebook --inplace --execute experiments/lqr/optimal.ipynb
echo optimal: 1 out of 1 run

# PBO linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

# PBO custom linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_custom_linear.ipynb
    echo PBO custom linear: $counter out of $N_SEEDS runs
    ((counter++))
done

# PBO optimal
jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_optimal.ipynb
echo PBO optimal: 1 out of 1 run
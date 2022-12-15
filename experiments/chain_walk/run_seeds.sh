#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/chain_walk/parameters.json -k max_bellman_iterations -v $N_BI


# FQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/FQI.ipynb
    echo FQI: $counter out of $N_SEEDS runs
    ((counter++))
done

# LSPI
jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/LSPI.ipynb
echo LSPI: 1 out of 1 run

# optimal
jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/optimal.ipynb
echo optimal: 1 out of 1 run

# PBO linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

# PBO max linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_max_linear.ipynb
    echo PBO max linear: $counter out of $N_SEEDS runs
    ((counter++))
done

# PBO optimal
jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_optimal.ipynb
echo PBO optimal: 1 out of 1 run
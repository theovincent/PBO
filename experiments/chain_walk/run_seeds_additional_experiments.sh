#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/chain_walk/parameters.json -k max_bellman_iterations -v $N_BI

# PBO linear
## n_weights = 20
edit_json -f experiments/chain_walk/parameters.json -k n_weights -v 20
edit_json -f experiments/chain_walk/parameters.json -k batch_size_weights -v 20

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

## n_weights = 50
edit_json -f experiments/chain_walk/parameters.json -k n_weights -v 50
edit_json -f experiments/chain_walk/parameters.json -k batch_size_weights -v 50

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

edit_json -f experiments/chain_walk/parameters.json -k n_weights -v 100
edit_json -f experiments/chain_walk/parameters.json -k batch_size_weights -v 100


# PBO max linear
## n_repetitions = 2
edit_json -f experiments/chain_walk/parameters.json -k n_repetitions -v 2

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_max_linear.ipynb
    echo PBO max linear: $counter out of $N_SEEDS runs
    ((counter++))
done

## n_repetitions = 5
edit_json -f experiments/chain_walk/parameters.json -k n_repetitions -v 5

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/chain_walk/PBO_max_linear.ipynb
    echo PBO max linear: $counter out of $N_SEEDS runs
    ((counter++))
done

edit_json -f experiments/chain_walk/parameters.json -k n_repetitions -v 10
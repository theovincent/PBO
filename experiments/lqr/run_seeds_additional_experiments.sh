#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/lqr/parameters.json -k max_bellman_iterations -v $N_BI


# PBO linear
## initial_weight_std = 1
edit_json -f experiments/lqr/parameters.json -k initial_weight_std -v 1

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

## initial_weight_std = 0.5
edit_json -f experiments/lqr/parameters.json -k initial_weight_std -v 0.5

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

edit_json -f experiments/lqr/parameters.json -k initial_weight_std -v 0.0005

# PBO custom linear
## n_discrete_states = 3, n_discrete_actions = 3
edit_json -f experiments/lqr/parameters.json -k n_discrete_states -v 3
edit_json -f experiments/lqr/parameters.json -k n_discrete_actions -v 3

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_custom_linear.ipynb
    echo PBO custom linear: $counter out of $N_SEEDS runs
    ((counter++))
done

## n_discrete_states = 5, n_discrete_actions = 5
edit_json -f experiments/lqr/parameters.json -k n_discrete_states -v 5
edit_json -f experiments/lqr/parameters.json -k n_discrete_actions -v 5

counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/lqr/PBO_custom_linear.ipynb
    echo PBO custom linear: $counter out of $N_SEEDS runs
    ((counter++))
done

edit_json -f experiments/lqr/parameters.json -k n_discrete_states -v 11
edit_json -f experiments/lqr/parameters.json -k n_discrete_actions -v 11
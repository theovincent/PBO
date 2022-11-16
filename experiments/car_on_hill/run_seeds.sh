#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

edit_json -f experiments/car_on_hill/parameters.json -k max_bellman_iterations -v $N_BI


# FQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/car_on_hill/FQI.ipynb
    echo FQI: $counter out of $N_SEEDS runs
    ((counter++))
done

# PBO linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/car_on_hill/PBO_linear.ipynb
    echo PBO linear: $counter out of $N_SEEDS runs
    ((counter++))
done

# PBO linear max linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute experiments/car_on_hill/PBO_linear_max_linear.ipynb
    echo PBO linear max linear: $counter out of $N_SEEDS runs
    ((counter++))
done
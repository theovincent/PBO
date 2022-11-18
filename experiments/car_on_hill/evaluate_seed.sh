#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

# PBO linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo PBO linear: $counter out of $N_SEEDS runs
    car_on_hill_pbo_evaluate -s $counter -b $N_BI -a linear
    ((counter++))
done

# PBO deep
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo PBO deep: $counter out of $N_SEEDS runs
    car_on_hill_pbo_evaluate -s $counter -b $N_BI -a deep
    ((counter++))
done

# IFQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo FQI: $counter out of $N_SEEDS runs
    car_on_hill_ifqi_evaluate -s $counter -b $N_BI
    ((counter++))
done
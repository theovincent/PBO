#!/bin/bash

source experiments/parse_arguments.sh
parse_arguments $@

# FQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo FQI: $counter out of $N_SEEDS runs
    car_on_hill_fqi -s $counter -b $N_BI
    ((counter++))
done

# PBO linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo PBO linear: $counter out of $N_SEEDS runs
    car_on_hill_pbo -s $counter -b $N_BI -a linear
    ((counter++))
done

# PBO deep
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo PBO deep: $counter out of $N_SEEDS runs
    car_on_hill_pbo -s $counter -b $N_BI -a linear
    ((counter++))
done

# IFQI
counter=1
while [ $counter -le  $N_SEEDS ]
do
    echo FQI: $counter out of $N_SEEDS runs
    car_on_hill_ifqi -s $counter -b $N_BI
    ((counter++))
done
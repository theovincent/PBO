#!/bin/bash

# FQI
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/FQI.ipynb
    echo FQI: $counter out of $1 runs
    ((counter++))
done

# LSPI
jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/LSPI.ipynb
echo LSPI: 1 out of 1 run

# optimal
jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/optimal.ipynb
echo optimal: 1 out of 1 run

# PBO linear
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/PBO_linear.ipynb
    echo PBO linear: $counter out of $1 runs
    ((counter++))
done

# PBO optimal
jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/PBO_optimal.ipynb
echo PBO optimal: 1 out of 1 run
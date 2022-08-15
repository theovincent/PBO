#!/bin/bash

# PBO linear
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/PBO_linear.ipynb
    echo PBO linear: $counter out of $1 runs
    ((counter++))
done

# PBO custom linear
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/PBO_custom_linear.ipynb
    echo PBO custom linear: $counter out of $1 runs
    ((counter++))
done
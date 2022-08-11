#!/bin/bash

# PBO max linear
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/PBO_linear.ipynb
    echo PBO max linear: $counter out of $1 runs
    ((counter++))
done
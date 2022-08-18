#!/bin/bash

# FQI
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/car_on_hill/FQI.ipynb
    echo FQI: $counter out of $1 runs
    ((counter++))
done
#!/bin/bash

counter=1

while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/sample_gap.ipynb
    echo $counter out of $1 runs
    ((counter++))
done
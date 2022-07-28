#!/bin/bash

counter=1

while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/PBO_max_linear.ipynb
    echo $counter out of $1 runs
    ((counter++))
done
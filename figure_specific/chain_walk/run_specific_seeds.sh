#!/bin/bash

# FQI
rm figure_specific/chain_walk/figures/data/FQI/*
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/FQI.ipynb
    echo FQI: $counter out of $1 runs
    ((counter++))
done
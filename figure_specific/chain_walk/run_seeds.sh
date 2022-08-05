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

# LSPI
rm figure_specific/chain_walk/figures/data/LSPI/*
jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/LSPI.ipynb
echo LSPI: 1 out of 1 run

# optimal
rm figure_specific/chain_walk/figures/data/optimal/*
jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/optimal.ipynb
echo optimal: 1 out of 1 run

# PBO linear
rm figure_specific/chain_walk/figures/data/PBO_linear/*
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/PBO_linear.ipynb
    echo PBO linear: $counter out of $1 runs
    ((counter++))
done

# PBO max linear
rm figure_specific/chain_walk/figures/data/PBO_max_linear/*
counter=1
while [ $counter -le  $1 ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/PBO_max_linear.ipynb
    echo PBO max linear: $counter out of $1 runs
    ((counter++))
done

# PBO optimal
rm figure_specific/chain_walk/figures/data/PBO_optimal/*
jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/PBO_optimal.ipynb
echo PBO optimal: 1 out of 1 run
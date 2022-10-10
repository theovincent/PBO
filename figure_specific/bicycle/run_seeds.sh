#!/bin/bash

source figure_specific/parse_arguments.sh
parse_arguments $@

edit_json -f figure_specific/bicycle/parameters.json -k max_bellman_iterations -v $N_BI


# # FQI
# counter=1
# while [ $counter -le  $N_SEEDS ]
# do
#     jupyter nbconvert --to notebook --inplace --execute figure_specific/bicycle/FQI.ipynb
#     echo FQI: $counter out of $N_SEEDS runs
#     ((counter++))
# done

# # PBO linear
# counter=1
# while [ $counter -le  $N_SEEDS ]
# do
#     jupyter nbconvert --to notebook --inplace --execute figure_specific/bicycle/PBO_linear.ipynb
#     echo PBO linear: $counter out of $N_SEEDS runs
#     ((counter++))
# done

# PBO linear max linear
counter=1
while [ $counter -le  $N_SEEDS ]
do
    jupyter nbconvert --to notebook --inplace --execute figure_specific/bicycle/PBO_linear_max_linear.ipynb
    echo PBO linear max linear: $counter out of $N_SEEDS runs
    ((counter++))
done
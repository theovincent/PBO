#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/lunar_lander/$EXPERIMENT_NAME ] || mkdir -p out/lunar_lander/$EXPERIMENT_NAME

[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lunar_lander/figures/$EXPERIMENT_NAME
[ -f experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lunar_lander/parameters.json experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/ProDQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/ProDQN


# ProDQN
echo "launch train prodqn"
submission_train_prodqn_1=$(sbatch -J L_$EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --ntasks=4 --mem-per-cpu=2G --time=06:00:00 --gres=gpu:1 -p gpu --output=out/lunar_lander/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_prodqn_%a.out launch_job/lunar_lander/train_prodqn.sh -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE)
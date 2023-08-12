#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/acrobot/$EXPERIMENT_NAME ] || mkdir -p out/acrobot/$EXPERIMENT_NAME

[ -d experiments/acrobot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/acrobot/figures/$EXPERIMENT_NAME
[ -f experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/acrobot/parameters.json experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/ProDQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/ProDQN


# ProDQN
echo "launch train prodqn"
submission_train_prodqn_1=$(sbatch -J A_$EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --ntasks=4 --mem-per-cpu=2G --time=03:00:00 --gres=gpu:1 -p gpu --output=out/acrobot/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_prodqn_%a.out launch_job/acrobot/train_prodqn.sh -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE)
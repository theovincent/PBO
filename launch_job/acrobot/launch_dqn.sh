#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/acrobot/$EXPERIMENT_NAME ] || mkdir -p out/acrobot/$EXPERIMENT_NAME

[ -d experiments/acrobot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/acrobot/figures/$EXPERIMENT_NAME
[ -f experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/acrobot/parameters.json experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/DQN


# DQN
echo "launch train dqn"
submission_train_dqn_1=$(sbatch -J A_$EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --ntasks=4 --mem-per-cpu=250M --time=01:00:00 --output=out/acrobot/$EXPERIMENT_NAME/train_dqn_%a.out launch_job/acrobot/train_dqn.sh -e $EXPERIMENT_NAME)
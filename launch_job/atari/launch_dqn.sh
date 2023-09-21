#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/DQN


# DQN
echo "launch train dqn"
submission_train_dqn_1=$(sbatch -J L_$EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=10G --time=5:00:00 --gres=gpu:1 -p gpu --output=out/atari/$EXPERIMENT_NAME/train_dqn_%a.out launch_job/atari/train_dqn.sh -e $EXPERIMENT_NAME)
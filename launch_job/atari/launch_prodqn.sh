#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/ProDQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/ProDQN

# ProDQN
echo "launch train prodqn"
submission_train_prodqn_1=$(sbatch -J A_$EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --ntasks=2 --mem-per-cpu=20G --time=1-10:00:00 --gres=gpu:1 -p gpu --output=out/atari/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_prodqn_%a.out launch_job/atari/train_prodqn.sh -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE)
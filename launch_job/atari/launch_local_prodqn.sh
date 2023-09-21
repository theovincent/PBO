#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/ProDQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/ProDQN

seed_command="export SLURM_ARRAY_TASK_ID=$FIRST_SEED"

# ProDQN
echo "launch train prodqn"
train_command="launch_job/atari/train_prodqn.sh -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE &> out/atari/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_prodqn_$FIRST_SEED.out"
tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER
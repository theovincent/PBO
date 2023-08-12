#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/acrobot/$EXPERIMENT_NAME ] || mkdir -p out/acrobot/$EXPERIMENT_NAME

[ -d experiments/acrobot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/acrobot/figures/$EXPERIMENT_NAME
[ -f experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/acrobot/parameters.json experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/DQN

seed_command="export SLURM_ARRAY_TASK_ID=$FIRST_SEED"

# DQN
echo "launch train dqn"
train_command="launch_job/acrobot/train_dqn.sh -e $EXPERIMENT_NAME &> out/acrobot/$EXPERIMENT_NAME/train_dqn_$FIRST_SEED.out"
tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER
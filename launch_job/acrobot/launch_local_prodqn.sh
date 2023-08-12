#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/acrobot/$EXPERIMENT_NAME ] || mkdir -p out/acrobot/$EXPERIMENT_NAME

[ -d experiments/acrobot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/acrobot/figures/$EXPERIMENT_NAME
[ -f experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/acrobot/parameters.json experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/ProDQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/ProDQN

seed_command="export SLURM_ARRAY_TASK_ID=$FIRST_SEED"

# ProDQN
echo "launch train prodqn"
train_command="launch_job/acrobot/train_prodqn.sh -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE &> out/acrobot/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_prodqn_$FIRST_SEED.out"
tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER
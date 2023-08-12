#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/lunar_lander/$EXPERIMENT_NAME ] || mkdir -p out/lunar_lander/$EXPERIMENT_NAME

[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lunar_lander/figures/$EXPERIMENT_NAME
[ -f experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lunar_lander/parameters.json experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/ProDQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/ProDQN

seed_command="export SLURM_ARRAY_TASK_ID=$FIRST_SEED"

# ProDQN
echo "launch train prodqn"
train_command="launch_job/lunar_lander/train_prodqn.sh -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE &> out/lunar_lander/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_prodqn_$FIRST_SEED.out"
tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER
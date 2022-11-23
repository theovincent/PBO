#!/bin/bash

source launch_job/parse_arguments.sh -e $EXPERIMENT_NAME
parse_arguments $@

mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME

echo "launch collect sample"
submission_collect_sample=$(sbatch -J collect_sample --mem-per-cpu=2Gc --time=00:50 --output=out/$EXPERIMENT_NAME/collect_sample.out --error=error/$EXPERIMENT_NAME/collect_sample.out launch_job/car_on_hill/collect_sample.sh -e $EXPERIMENT_NAME)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}

echo "launch train and evaluate" 
submission_train_and_evaluate=$(sbatch -J train_and_evaluate --dependency=afterok:$submission_id_collect_sample --mem-per-cpu=20Mc --time=00:10 --output=out/$EXPERIMENT_NAME/train_and_evaluate.out --error=error/$EXPERIMENT_NAME/train_and_evaluate.out launch_job/car_on_hill/train_and_evaluate.sh -e $EXPERIMENT_NAME -fs $FIRST_SEED -ls $LAST_SEED -b $MAX_BELLMAN_ITERATION)
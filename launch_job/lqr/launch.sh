#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/lqr/$EXPERIMENT_NAME ] || mkdir -p out/lqr/$EXPERIMENT_NAME
[ -d error/lqr/$EXPERIMENT_NAME ] || mkdir -p error/lqr/$EXPERIMENT_NAME

[ -d experiments/lqr/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lqr/figures/$EXPERIMENT_NAME
[ -f experiments/lqr/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lqr/parameters.json experiments/lqr/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/LSPI ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/LSPI
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_custom_linear ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_custom_linear
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/lqr/figures/$EXPERIMENT_NAME/PBO_optimal ] || mkdir experiments/lqr/figures/$EXPERIMENT_NAME/PBO_optimal


# Collect data
echo "launch collect sample"
submission_collect_sample=$(sbatch -J l_collect_sample --cpus-per-task=2 --mem-per-cpu=100Mc --time=5:00 --output=out/lqr/$EXPERIMENT_NAME/collect_sample.out --error=error/lqr/$EXPERIMENT_NAME/collect_sample.out -p amd,amd2 launch_job/lqr/collect_sample.sh -e $EXPERIMENT_NAME -b 0)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}


if [[ $FQI = true ]]
then
    # FQI
    echo "launch train fqi"
    submission_train_fqi=$(sbatch -J l_train_fqi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=200Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/train_fqi_%a.out --error=error/lqr/$EXPERIMENT_NAME/train_fqi_%a.out -p amd,amd2 launch_job/lqr/train_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_fqi <<< $submission_train_fqi
    submission_id_train_fqi=${split_submission_train_fqi[-1]}

    echo "launch evaluate fqi"
    submission_evaluate_fqi=$(sbatch -J l_evaluate_fqi --dependency=afterok:$submission_id_train_fqi,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=200Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/evaluate_fqi_%a.out --error=error/lqr/$EXPERIMENT_NAME/evaluate_fqi_%a.out -p amd,amd2 launch_job/lqr/evaluate_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 

if [[ $LSPI = true ]]
then
    # LSPI
    echo "launch train lspi"
    submission_train_lspi=$(sbatch -J l_train_lspi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=200Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/train_lspi.out --error=error/lqr/$EXPERIMENT_NAME/train_lspi.out -p amd,amd2 launch_job/lqr/train_lspi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi


if [[ $PBO_linear = true ]]
then
    # PBO linear
    echo "launch train pbo linear"
    submission_train_pbo_linear=$(sbatch -J l_train_pbo_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=200Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/train_pbo_linear_%a.out --error=error/lqr/$EXPERIMENT_NAME/train_pbo_linear_%a.out -p amd,amd2 launch_job/lqr/train_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear"
    submission_evaluate_pbo_linear=$(sbatch -J l_evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=300Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out --error=error/lqr/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out -p amd,amd2 launch_job/lqr/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)
fi


if [[ $PBO_deep = true ]]
then
    # PBO deep
    echo "launch train pbo deep"
    submission_train_pbo_deep=$(sbatch -J l_train_pbo_deep --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=200Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/train_pbo_deep_%a.out --error=error/lqr/$EXPERIMENT_NAME/train_pbo_deep_%a.out -p amd,amd2 launch_job/lqr/train_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep"
    submission_evaluate_pbo_deep=$(sbatch -J l_evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=600Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out --error=error/lqr/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out -p amd,amd2 launch_job/lqr/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep)
fi


if [[ $PBO_custom_linear = true ]]
then
    # PBO max linear
    echo "launch train pbo custom_linear"
    submission_train_pbo_custom_linear=$(sbatch -J l_train_pbo_custom_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=200Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/train_pbo_custom_linear_%a.out --error=error/lqr/$EXPERIMENT_NAME/train_pbo_custom_linear_%a.out -p amd,amd2 launch_job/lqr/train_pbo_custom_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a custom_linear)

    IFS=" " read -ra split_submission_train_pbo_custom_linear <<< $submission_train_pbo_custom_linear
    submission_id_train_pbo_custom_linear=${split_submission_train_pbo_custom_linear[-1]}

    echo "launch evaluate pbo custom_linear"
    submission_evaluate_pbo_custom_linear=$(sbatch -J l_evaluate_pbo_custom_linear --dependency=afterok:$submission_id_train_pbo_custom_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=600Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/evaluate_pbo_custom_linear_%a.out --error=error/lqr/$EXPERIMENT_NAME/evaluate_pbo_custom_linear_%a.out -p amd,amd2 launch_job/lqr/evaluate_pbo_custom_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a custom_linear)
fi


if [[ $PBO_optimal = true ]]
then
    # PBO optimal
    echo "launch evaluate pbo optimal"
    submission_train_pbo_optimal=$(sbatch -J l_evaluate_pbo_optimal --dependency=afterok:$submission_id_collect_sample --cpus-per-task=2 --mem-per-cpu=600Mc --time=20:00 --output=out/lqr/$EXPERIMENT_NAME/evaluate_pbo_optimal.out --error=error/lqr/$EXPERIMENT_NAME/evaluate_pbo_optimal.out -p amd,amd2 launch_job/lqr/evaluate_pbo_optimal.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi
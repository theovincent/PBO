#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/chain_walk/$EXPERIMENT_NAME ] || mkdir -p out/chain_walk/$EXPERIMENT_NAME
[ -d error/chain_walk/$EXPERIMENT_NAME ] || mkdir -p error/chain_walk/$EXPERIMENT_NAME

[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/chain_walk/figures/$EXPERIMENT_NAME
[ -f experiments/chain_walk/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/chain_walk/parameters.json experiments/chain_walk/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/LSPI ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/LSPI
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_max_linear ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_max_linear
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_optimal ] || mkdir experiments/chain_walk/figures/$EXPERIMENT_NAME/PBO_optimal


# Collect data
echo "launch collect sample"
submission_collect_sample=$(sbatch -J c_collect_sample --cpus-per-task=3 --mem-per-cpu=500Mc --time=5:00 --output=out/chain_walk/$EXPERIMENT_NAME/collect_sample.out --error=error/chain_walk/$EXPERIMENT_NAME/collect_sample.out -p amd,amd2 launch_job/chain_walk/collect_sample.sh -e $EXPERIMENT_NAME -b 0)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}


if [[ $FQI = true ]]
then
    # FQI
    echo "launch train fqi"
    submission_train_fqi=$(sbatch -J c_train_fqi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=30:00 --output=out/chain_walk/$EXPERIMENT_NAME/train_fqi_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/train_fqi_%a.out -p amd,amd2 launch_job/chain_walk/train_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_fqi <<< $submission_train_fqi
    submission_id_train_fqi=${split_submission_train_fqi[-1]}

    echo "launch evaluate fqi"
    submission_evaluate_fqi=$(sbatch -J c_evaluate_fqi --dependency=afterok:$submission_id_train_fqi,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=200Mc --time=10:00 --output=out/chain_walk/$EXPERIMENT_NAME/evaluate_fqi_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/evaluate_fqi_%a.out -p amd,amd2 launch_job/chain_walk/evaluate_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 

if [[ $LSPI = true ]]
then
    # LSPI
    echo "launch train lspi"
    submission_train_lspi=$(sbatch -J c_train_lspi --dependency=afterok:$submission_id_collect_sample --cpus-per-task=3 --mem-per-cpu=750Mc --time=30:00 --output=out/chain_walk/$EXPERIMENT_NAME/train_lspi.out --error=error/chain_walk/$EXPERIMENT_NAME/train_lspi.out -p amd,amd2 launch_job/chain_walk/train_lspi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi


if [[ $PBO_linear = true ]]
then
    # PBO linear
    echo "launch train pbo linear"
    submission_train_pbo_linear=$(sbatch -J c_train_pbo_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=30:00 --output=out/chain_walk/$EXPERIMENT_NAME/train_pbo_linear_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/train_pbo_linear_%a.out -p amd,amd2 launch_job/chain_walk/train_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear"
    submission_evaluate_pbo_linear=$(sbatch -J c_evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=300Mc --time=10:00 --output=out/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out -p amd,amd2 launch_job/chain_walk/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)
fi


if [[ $PBO_deep = true ]]
then
    # PBO deep
    echo "launch train pbo deep"
    submission_train_pbo_deep=$(sbatch -J c_train_pbo_deep --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=30:00 --output=out/chain_walk/$EXPERIMENT_NAME/train_pbo_deep_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/train_pbo_deep_%a.out -p amd,amd2 launch_job/chain_walk/train_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep"
    submission_evaluate_pbo_deep=$(sbatch -J c_evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=600Mc --time=10:00 --output=out/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out -p amd,amd2 launch_job/chain_walk/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep)
fi


if [[ $PBO_max_linear = true ]]
then
    # PBO max linear
    echo "launch train pbo max_linear"
    submission_train_pbo_max_linear=$(sbatch -J c_train_pbo_max_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=30:00 --output=out/chain_walk/$EXPERIMENT_NAME/train_pbo_max_linear_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/train_pbo_max_linear_%a.out -p amd,amd2 launch_job/chain_walk/train_pbo_max_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a max_linear)

    IFS=" " read -ra split_submission_train_pbo_max_linear <<< $submission_train_pbo_max_linear
    submission_id_train_pbo_max_linear=${split_submission_train_pbo_max_linear[-1]}

    echo "launch evaluate pbo max_linear"
    submission_evaluate_pbo_max_linear=$(sbatch -J c_evaluate_pbo_max_linear --dependency=afterok:$submission_id_train_pbo_max_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=600Mc --time=10:00 --output=out/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_max_linear_%a.out --error=error/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_max_linear_%a.out -p amd,amd2 launch_job/chain_walk/evaluate_pbo_max_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a max_linear)
fi

if [[ $PBO_optimal = true ]]
then
    # PBO optimal
    echo "launch evaluate pbo optimal"
    submission_train_pbo_optimal=$(sbatch -J c_evaluate_pbo_optimal --dependency=afterok:$submission_id_collect_sample --cpus-per-task=3 --mem-per-cpu=600Mc --time=10:00 --output=out/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_optimal.out --error=error/chain_walk/$EXPERIMENT_NAME/evaluate_pbo_optimal.out -p amd,amd2 launch_job/chain_walk/evaluate_pbo_optimal.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi
#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/car_on_hill/$EXPERIMENT_NAME ] || mkdir -p out/car_on_hill/$EXPERIMENT_NAME
[ -d error/car_on_hill/$EXPERIMENT_NAME ] || mkdir -p error/car_on_hill/$EXPERIMENT_NAME

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -f experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/PBO_deep


# Collect data
echo "launch collect sample"
submission_collect_sample=$(sbatch -J C_collect_sample --cpus-per-task=3 --mem-per-cpu=500Mc --time=50:00 --output=out/car_on_hill/$EXPERIMENT_NAME/collect_sample.out --error=error/car_on_hill/$EXPERIMENT_NAME/collect_sample.out -p amd,amd2 launch_job/car_on_hill/collect_sample.sh -e $EXPERIMENT_NAME -b 0)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}


if [[ $FQI = true ]]
then
    # FQI
    echo "launch train fqi"
    submission_train_fqi=$(sbatch -J C_train_fqi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=1:30:00 --output=out/car_on_hill/$EXPERIMENT_NAME/train_fqi_%a.out --error=error/car_on_hill/$EXPERIMENT_NAME/train_fqi_%a.out -p amd,amd2 launch_job/car_on_hill/train_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_fqi <<< $submission_train_fqi
    submission_id_train_fqi=${split_submission_train_fqi[-1]}

    echo "launch evaluate fqi"
    submission_evaluate_fqi=$(sbatch -J C_evaluate_fqi --dependency=afterok:$submission_id_train_fqi,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=100Mc --time=10:00 --output=out/car_on_hill/$EXPERIMENT_NAME/evaluate_fqi_%a.out --error=error/car_on_hill/$EXPERIMENT_NAME/evaluate_fqi_%a.out -p amd,amd2 launch_job/car_on_hill/evaluate_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 


if [[ $PBO_linear = true ]]
then
    # PBO linear
    echo "launch train pbo linear"
    submission_train_pbo_linear=$(sbatch -J C_train_pbo_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=3:30:00 --output=out/car_on_hill/$EXPERIMENT_NAME/train_pbo_linear_%a.out --error=error/car_on_hill/$EXPERIMENT_NAME/train_pbo_linear_%a.out -p amd,amd2 launch_job/car_on_hill/train_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear"
    submission_evaluate_pbo_linear=$(sbatch -J C_evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=300Mc --time=10:00 --output=out/car_on_hill/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out --error=error/car_on_hill/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out -p amd,amd2 launch_job/car_on_hill/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)
fi


if [[ $PBO_deep = true ]]
then
    # PBO deep
    echo "launch train pbo deep"
    submission_train_pbo_deep=$(sbatch -J C_train_pbo_deep --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=15:30:00 --output=out/car_on_hill/$EXPERIMENT_NAME/train_pbo_deep_%a.out --error=error/car_on_hill/$EXPERIMENT_NAME/train_pbo_deep_%a.out -p amd,amd2 launch_job/car_on_hill/train_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep"
    submission_evaluate_pbo_deep=$(sbatch -J C_evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=600Mc --time=10:00 --output=out/car_on_hill/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out --error=error/car_on_hill/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out -p amd,amd2 launch_job/car_on_hill/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV)
fi
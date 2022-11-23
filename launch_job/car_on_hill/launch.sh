#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

mkdir out/$EXPERIMENT_NAME
mkdir error/$EXPERIMENT_NAME

echo "launch collect sample"
submission_collect_sample=$(sbatch -J collect_sample --mem-per-cpu=2Gc --time=00:50 --output=out/$EXPERIMENT_NAME/collect_sample.out --error=error/$EXPERIMENT_NAME/collect_sample.out launch_job/car_on_hill/collect_sample.sh -e $EXPERIMENT_NAME -s 0 -b 0)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do  
    # FQI
    echo "launch train fqi for seed" $seed 
    submission_train_fqi=$(sbatch -J train_fqi --dependency=afterok:$submission_id_collect_sample --mem-per-cpu=8Gc --time=10:00 --output=out/$EXPERIMENT_NAME/train_fqi_$seed.out --error=error/$EXPERIMENT_NAME/train_fqi_$seed.out launch_job/car_on_hill/train_fqi.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_fqi <<< $submission_train_fqi
    submission_id_train_fqi=${split_submission_train_fqi[-1]}

    echo "launch evaluate fqi for seed" $seed 
    submission_evaluate_fqi=$(sbatch -J evaluate_fqi --dependency=afterok:$submission_id_train_fqi,$submission_id_collect_sample --mem-per-cpu=4Gc --time=10:00 --output=out/$EXPERIMENT_NAME/evaluate_fqi_$seed.out --error=error/$EXPERIMENT_NAME/evaluate_fqi_$seed.out launch_job/car_on_hill/evaluate_fqi.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION)


    # PBO linear
    echo "launch train pbo linear for seed" $seed 
    submission_train_pbo_linear=$(sbatch -J train_pbo_linear --dependency=afterok:$submission_id_collect_sample --mem-per-cpu=8Gc --time=10:00 --output=out/$EXPERIMENT_NAME/train_pbo_linear_$seed.out --error=error/$EXPERIMENT_NAME/train_pbo_linear_$seed.out launch_job/car_on_hill/train_pbo_linear.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION -a linear)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear for seed" $seed 
    submission_evaluate_pbo_linear=$(sbatch -J evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear,$submission_id_collect_sample --mem-per-cpu=4Gc --time=10:00 --output=out/$EXPERIMENT_NAME/evaluate_pbo_linear_$seed.out --error=error/$EXPERIMENT_NAME/evaluate_pbo_linear_$seed.out launch_job/car_on_hill/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION -a linear)


    # PBO deep
    echo "launch train pbo deep for seed" $seed 
    submission_train_pbo_deep=$(sbatch -J train_pbo_deep --dependency=afterok:$submission_id_collect_sample --mem-per-cpu=8Gc --time=10:00 --output=out/$EXPERIMENT_NAME/train_pbo_deep_$seed.out --error=error/$EXPERIMENT_NAME/train_pbo_deep_$seed.out launch_job/car_on_hill/train_pbo_deep.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION -a deep)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep for seed" $seed 
    submission_evaluate_pbo_deep=$(sbatch -J evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep,$submission_id_collect_sample --mem-per-cpu=4Gc --time=10:00 --output=out/$EXPERIMENT_NAME/evaluate_pbo_deep_$seed.out --error=error/$EXPERIMENT_NAME/evaluate_pbo_deep_$seed.out launch_job/car_on_hill/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION -a deep)


    # IFQI
    echo "launch train ifqi for seed" $seed 
    submission_train_ifqi=$(sbatch -J train_ifqi --dependency=afterok:$submission_id_collect_sample --mem-per-cpu=8Gc --time=10:00 --output=out/$EXPERIMENT_NAME/train_ifqi_$seed.out --error=error/$EXPERIMENT_NAME/train_ifqi_$seed.out launch_job/car_on_hill/train_ifqi.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_ifqi <<< $submission_train_ifqi
    submission_id_train_ifqi=${split_submission_train_ifqi[-1]}

    echo "launch evaluate ifqi for seed" $seed 
    submission_evaluate_ifqi=$(sbatch -J evaluate_ifqi --dependency=afterok:$submission_id_train_ifqi,$submission_id_collect_sample --mem-per-cpu=4Gc --time=10:00 --output=out/$EXPERIMENT_NAME/evaluate_ifqi_$seed.out --error=error/$EXPERIMENT_NAME/evaluate_ifqi_$seed.out launch_job/car_on_hill/evaluate_ifqi.sh -e $EXPERIMENT_NAME -s $seed -b $MAX_BELLMAN_ITERATION)
done
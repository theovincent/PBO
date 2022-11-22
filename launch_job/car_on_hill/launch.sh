#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if [[ $COLLECT_SAMPLE = true ]]
then
    sbatch -J collect_sample --mem-per-cpu=2Gc --time=00:50 --output=out/collect_sample.out launch_job/car_on_hill/collect_sample.sh 
fi


submission_train_fqi=$(sbatch -J train_fqi --mem-per-cpu=8Gc --time=04:30 --gpus=1 --output=out/train_fqi_$seed.out launch_job/car_on_hill/train_fqi.sh -s $SEED -b $MAX_BELLMAN_ITERATION)

submission_train_pbo_linear=$(sbatch -J train_pbo_linear --mem-per-cpu=8Gc --time=02:30 --gpus=1 --output=out/train_pbo_linear_$seed.out launch_job/car_on_hill/train_pbo_linear.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a linear)
submission_evaluate_pbo_linear=$(sbatch -J evaluate_pbo_linear --dependency=afterok:$submission_train_pbo_linear --mem-per-cpu=4Gc --time=02:30 --output=out/evaluate_pbo_linear_$seed.out launch_job/car_on_hill/evaluate_pbo_linear.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a linear)

submission_train_pbo_deep=$(sbatch -J train_pbo_deep --mem-per-cpu=8Gc --time=02:30 --gpus=1 --output=out/train_pbo_deep_$seed.out launch_job/car_on_hill/train_pbo_deep.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a deep)
submission_evaluate_pbo_deep=$(sbatch -J evaluate_pbo_deep --dependency=afterok:$submission_train_pbo_deep --mem-per-cpu=4Gc --time=02:30 --output=out/evaluate_pbo_deep_$seed.out launch_job/car_on_hill/evaluate_pbo_deep.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a deep)

submission_train_ifqi=$(sbatch -J train_ifqi --mem-per-cpu=8Gc --time=02:30 --gpus=1 --output=out/train_ifqi_$seed.out launch_job/car_on_hill/train_ifqi.sh -s $SEED -b $MAX_BELLMAN_ITERATION)
submission_evaluate_ifqi=$(sbatch -J evaluate_ifqi --dependency=afterok:$submission_train_ifqi --mem-per-cpu=4Gc --time=02:30 --output=out/evaluate_ifqi_$seed.out launch_job/car_on_hill/evaluate_ifqi.sh -s $SEED -b $MAX_BELLMAN_ITERATION)

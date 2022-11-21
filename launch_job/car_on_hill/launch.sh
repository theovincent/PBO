#!/bin/bash

if collect_sample:
    sbatch -J collect_sample --mem=2Gc --time=00:50 --output=out/collect_sample.out launch_job/car_on_hill/collect_sample.sh 


for seed in SEEDS:
    ASK FOR GPU SMALL N CORES submission_train=$(sbatch -J train_fqi --mem=8Gc --time=04:30 --output=out/train_fqi_$seed.out launch_job/car_on_hill/train_fqi.sh -s $SEED -b $MAX_BELLMAN_ITERATION)
   
    submission_train=$(sbatch -J train_pbo_linear --mem=8Gc --time=02:30 --output=out/train_pbo_linear_$seed.out launch_job/car_on_hill/train_pbo_linear.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a linear)
    ASK FOR CPU BIG N CORES submission_evaluate=$(sbatch -J evaluate_pbo_linear --dependency=afterany:$submission_train --mem=4Gc --time=02:30 --output=out/evaluate_pbo_linear_$seed.out launch_job/car_on_hill/evaluate_pbo_linear.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a linear)

    submission_train=$(sbatch -J train_pbo_deep --mem=8Gc --time=02:30 --output=out/train_pbo_deep_$seed.out launch_job/car_on_hill/train_pbo_deep.sh -s $SEED -b $MAX_BELLMAN_ITERATION -a deep)
    submission_train=$(sbatch -J train_ifqi --mem=8Gc --time=02:30 --output=out/train_ifqi_$seed.out launch_job/car_on_hill/train_ifqi.sh -s $SEED -b $MAX_BELLMAN_ITERATION)


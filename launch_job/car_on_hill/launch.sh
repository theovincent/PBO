#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if [[ $COLLECT_SAMPLE = true ]]
then
    echo launch collect sample
    sbatch -J collect_sample --mem-per-cpu=2Gc --time=00:50 --output=out/collect_sample.out launch_job/car_on_hill/collect_sample.sh 
fi


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
	echo "launch train fqi for seed " $seed
    submission_train_fqi=$(sbatch -J train_fqi --mem-per-cpu=8Gc --time=04:30 --output=out/train_fqi_$seed.out launch_job/car_on_hill/train_fqi.sh -s $seed -b $MAX_BELLMAN_ITERATION)


    echo "launch train pbo linear for seed" $seed 
    submission_train_pbo_linear=$(sbatch -J train_pbo_linear --mem-per-cpu=8Gc --time=02:30 --output=out/train_pbo_linear_$seed.out launch_job/car_on_hill/train_pbo_linear.sh -s $seed -b $MAX_BELLMAN_ITERATION -a linear)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear for seed" $seed 
    submission_evaluate_pbo_linear=$(sbatch -J evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear --mem-per-cpu=4Gc --time=02:30 --output=out/evaluate_pbo_linear_$seed.out launch_job/car_on_hill/evaluate_pbo_linear.sh -s $seed -b $MAX_BELLMAN_ITERATION -a linear)


    echo "launch train pbo deep for seed" $seed 
    submission_train_pbo_deep=$(sbatch -J train_pbo_deep --mem-per-cpu=8Gc --time=02:30 --output=out/train_pbo_deep_$seed.out launch_job/car_on_hill/train_pbo_deep.sh -s $seed -b $MAX_BELLMAN_ITERATION -a deep)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep for seed" $seed 
    submission_evaluate_pbo_deep=$(sbatch -J evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep --mem-per-cpu=4Gc --time=02:30 --output=out/evaluate_pbo_deep_$seed.out launch_job/car_on_hill/evaluate_pbo_deep.sh -s $seed -b $MAX_BELLMAN_ITERATION -a deep)


    echo "launch train ifqi for seed" $seed 
    submission_train_ifqi=$(sbatch -J train_ifqi --mem-per-cpu=8Gc --time=02:30 --output=out/train_ifqi_$seed.out launch_job/car_on_hill/train_ifqi.sh -s $seed -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_ifqi <<< $submission_train_ifqi
    submission_id_train_ifqi=${split_submission_train_ifqi[-1]}

    echo "launch evaluate ifqi for seed" $seed 
    submission_evaluate_ifqi=$(sbatch -J evaluate_ifqi --dependency=afterok:$submission_id_train_ifqi --mem-per-cpu=4Gc --time=02:30 --output=out/evaluate_ifqi_$seed.out launch_job/car_on_hill/evaluate_ifqi.sh -s $seed -b $MAX_BELLMAN_ITERATION)

done

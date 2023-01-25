#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/bicycle_offline/$EXPERIMENT_NAME ] || mkdir -p out/bicycle_offline/$EXPERIMENT_NAME
[ -d error/bicycle_offline/$EXPERIMENT_NAME ] || mkdir -p error/bicycle_offline/$EXPERIMENT_NAME

[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/bicycle_offline/figures/$EXPERIMENT_NAME
[ -f experiments/bicycle_offline/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/bicycle_offline/parameters.json experiments/bicycle_offline/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/FQI ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/FQI
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/bicycle_offline/figures/$EXPERIMENT_NAME/IFQI ] || mkdir experiments/bicycle_offline/figures/$EXPERIMENT_NAME/IFQI


# Collect data
echo "launch collect sample"
submission_collect_sample=$(sbatch -J Boff_collect_sample --cpus-per-task=3 --mem-per-cpu=250Mc --time=50:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/collect_sample.out --error=error/bicycle_offline/$EXPERIMENT_NAME/collect_sample.out -p amd,amd2 launch_job/bicycle_offline/collect_sample.sh -e $EXPERIMENT_NAME -b 0)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}


if [[ $FQI = true ]]
then
    # FQI
    echo "launch train fqi"
    submission_train_fqi=$(sbatch -J Boff_train_fqi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=1:30:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/train_fqi_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/train_fqi_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/bicycle_offline/train_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -g)

    IFS=" " read -ra split_submission_train_fqi <<< $submission_train_fqi
    submission_id_train_fqi=${split_submission_train_fqi[-1]}

    echo "launch evaluate fqi"
    submission_evaluate_fqi=$(sbatch -J Boff_evaluate_fqi --dependency=afterok:$submission_id_train_fqi,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=5 --mem-per-cpu=700Mc --time=20:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/evaluate_fqi_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/evaluate_fqi_%a.out -p amd,amd2 launch_job/bicycle_offline/evaluate_fqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 


if [[ $PBO_linear = true ]]
then
    # PBO linear
    echo "launch train pbo linear"
    submission_train_pbo_linear=$(sbatch -J Boff_train_pbo_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=3:30:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/train_pbo_linear_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/train_pbo_linear_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/bicycle_offline/train_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear -g)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear"
    submission_evaluate_pbo_linear=$(sbatch -J Boff_evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=5 --mem-per-cpu=700Mc --time=20:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out -p amd,amd2 launch_job/bicycle_offline/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)
fi


if [[ $PBO_deep = true ]]
then
    # PBO deep
    echo "launch train pbo deep"
    submission_train_pbo_deep=$(sbatch -J Boff_train_pbo_deep --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=7:30:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/train_pbo_deep_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/train_pbo_deep_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/bicycle_offline/train_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV -g)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep"
    submission_evaluate_pbo_deep=$(sbatch -J Boff_evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=5 --mem-per-cpu=700Mc --time=20:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out -p amd,amd2 launch_job/bicycle_offline/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV)
fi


if [[ $IFQI = true ]]
then
    # IFQI
    echo "launch train ifqi"
    submission_train_ifqi=$(sbatch -J Boff_train_ifqi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=3:30:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/train_ifqi_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/train_ifqi_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/bicycle_offline/train_ifqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -g)

    IFS=" " read -ra split_submission_train_ifqi <<< $submission_train_ifqi
    submission_id_train_ifqi=${split_submission_train_ifqi[-1]}

    echo "launch evaluate ifqi"
    submission_evaluate_ifqi=$(sbatch -J Boff_evaluate_ifqi --dependency=afterok:$submission_id_train_ifqi,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=5 --mem-per-cpu=700Mc --time=20:00 --output=out/bicycle_offline/$EXPERIMENT_NAME/evaluate_ifqi_%a.out --error=error/bicycle_offline/$EXPERIMENT_NAME/evaluate_ifqi_%a.out -p amd,amd2 launch_job/bicycle_offline/evaluate_ifqi.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi
#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/bicycle_online/$EXPERIMENT_NAME ] || mkdir -p out/bicycle_online/$EXPERIMENT_NAME
[ -d error/bicycle_online/$EXPERIMENT_NAME ] || mkdir -p error/bicycle_online/$EXPERIMENT_NAME

[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/bicycle_online/figures/$EXPERIMENT_NAME
[ -f experiments/bicycle_online/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/bicycle_online/parameters.json experiments/bicycle_online/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/DQN
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/bicycle_online/figures/$EXPERIMENT_NAME/IDQN ] || mkdir experiments/bicycle_online/figures/$EXPERIMENT_NAME/IDQN


# Collect data
echo "launch collect sample"
submission_collect_sample=$(sbatch -J Bon_collect_sample --cpus-per-task=3 --mem-per-cpu=250Mc --time=50:00 --output=out/bicycle_online/$EXPERIMENT_NAME/collect_sample.out --error=error/bicycle_online/$EXPERIMENT_NAME/collect_sample.out -p amd,amd2 launch_job/bicycle_online/collect_sample.sh -e $EXPERIMENT_NAME -b 0)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}


if [[ $DQN = true ]]
then
    # DQN
    echo "launch train dqn"
    submission_train_dqn=$(sbatch -J Bon_train_dqn --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=15:30:00 --output=out/bicycle_online/$EXPERIMENT_NAME/train_dqn_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/train_dqn_%a.out -p amd,amd2 launch_job/bicycle_online/train_dqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_dqn <<< $submission_train_dqn
    submission_id_train_dqn=${split_submission_train_dqn[-1]}

    echo "launch evaluate dqn"
    submission_evaluate_dqn=$(sbatch -J Bon_evaluate_dqn --dependency=afterok:$submission_id_train_dqn,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=100Mc --time=10:00 --output=out/bicycle_online/$EXPERIMENT_NAME/evaluate_dqn_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/evaluate_dqn_%a.out -p amd,amd2 launch_job/bicycle_online/evaluate_dqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 


if [[ $PBO_linear = true ]]
then
    # PBO linear
    echo "launch train pbo linear"
    submission_train_pbo_linear=$(sbatch -J Bon_train_pbo_linear --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=15:30:00 --output=out/bicycle_online/$EXPERIMENT_NAME/train_pbo_linear_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/train_pbo_linear_%a.out -p amd,amd2 launch_job/bicycle_online/train_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear"
    submission_evaluate_pbo_linear=$(sbatch -J Bon_evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=300Mc --time=10:00 --output=out/bicycle_online/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out -p amd,amd2 launch_job/bicycle_online/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)
fi


if [[ $PBO_deep = true ]]
then
    # PBO deep
    echo "launch train pbo deep"
    submission_train_pbo_deep=$(sbatch -J Bon_train_pbo_deep --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=17:30:00 --output=out/bicycle_online/$EXPERIMENT_NAME/train_pbo_deep_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/train_pbo_deep_%a.out -p amd,amd2 launch_job/bicycle_online/train_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep"
    submission_evaluate_pbo_deep=$(sbatch -J Bon_evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=600Mc --time=10:00 --output=out/bicycle_online/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out -p amd,amd2 launch_job/bicycle_online/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV)
fi


if [[ $IDQN = true ]]
then
    # IDQN
    echo "launch train idqn"
    submission_train_idqn=$(sbatch -J Bon_train_idqn --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=15:30:00 --output=out/bicycle_online/$EXPERIMENT_NAME/train_idqn_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/train_idqn_%a.out -p amd,amd2 launch_job/bicycle_online/train_idqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)

    IFS=" " read -ra split_submission_train_idqn <<< $submission_train_idqn
    submission_id_train_idqn=${split_submission_train_idqn[-1]}

    echo "launch evaluate idqn"
    submission_evaluate_idqn=$(sbatch -J Bon_evaluate_idqn --dependency=afterok:$submission_id_train_idqn,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=150Mc --time=10:00 --output=out/bicycle_online/$EXPERIMENT_NAME/evaluate_idqn_%a.out --error=error/bicycle_online/$EXPERIMENT_NAME/evaluate_idqn_%a.out -p amd,amd2 launch_job/bicycle_online/evaluate_idqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi
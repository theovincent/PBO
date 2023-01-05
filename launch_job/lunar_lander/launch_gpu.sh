#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/lunar_lander/$EXPERIMENT_NAME ] || mkdir -p out/lunar_lander/$EXPERIMENT_NAME
[ -d error/lunar_lander/$EXPERIMENT_NAME ] || mkdir -p error/lunar_lander/$EXPERIMENT_NAME

[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lunar_lander/figures/$EXPERIMENT_NAME
[ -f experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lunar_lander/parameters.json experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/DQN
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_linear ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_linear
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_deep ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/PBO_deep
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/IDQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/IDQN


if [[ $DQN = true ]]
then
    # DQN
    echo "launch train dqn"
    submission_train_dqn=$(sbatch -J train_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=750Mc --time=3:30:00 --output=out/lunar_lander/$EXPERIMENT_NAME/train_dqn_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/train_dqn_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/lunar_lander/train_dqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -g)

    IFS=" " read -ra split_submission_train_dqn <<< $submission_train_dqn
    submission_id_train_dqn=${split_submission_train_dqn[-1]}

    echo "launch evaluate dqn"
    submission_evaluate_dqn=$(sbatch -J evaluate_dqn --dependency=afterok:$submission_id_train_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=4000Mc --time=15:00 --output=out/lunar_lander/$EXPERIMENT_NAME/evaluate_dqn_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/evaluate_dqn_%a.out -p amd,amd2 launch_job/lunar_lander/evaluate_dqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 


if [[ $PBO_linear = true ]]
then
    # PBO linear
    echo "launch train pbo linear"
    submission_train_pbo_linear=$(sbatch -J train_pbo_linear --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=750Mc --time=3:30:00 --output=out/lunar_lander/$EXPERIMENT_NAME/train_pbo_linear_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/train_pbo_linear_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/lunar_lander/train_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear -g)

    IFS=" " read -ra split_submission_train_pbo_linear <<< $submission_train_pbo_linear
    submission_id_train_pbo_linear=${split_submission_train_pbo_linear[-1]}

    echo "launch evaluate pbo linear"
    submission_evaluate_pbo_linear=$(sbatch -J evaluate_pbo_linear --dependency=afterok:$submission_id_train_pbo_linear --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=4000Mc --time=15:00 --output=out/lunar_lander/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/evaluate_pbo_linear_%a.out -p amd,amd2 launch_job/lunar_lander/evaluate_pbo_linear.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a linear)
fi


if [[ $PBO_deep = true ]]
then
    # PBO deep
    echo "launch train pbo deep"
    submission_train_pbo_deep=$(sbatch -J train_pbo_deep --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=1200Mc --time=5:30:00 --output=out/lunar_lander/$EXPERIMENT_NAME/train_pbo_deep_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/train_pbo_deep_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/lunar_lander/train_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV -g)

    IFS=" " read -ra split_submission_train_pbo_deep <<< $submission_train_pbo_deep
    submission_id_train_pbo_deep=${split_submission_train_pbo_deep[-1]}

    echo "launch evaluate pbo deep"
    submission_evaluate_pbo_deep=$(sbatch -J evaluate_pbo_deep --dependency=afterok:$submission_id_train_pbo_deep --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=4000Mc --time=15:00 --output=out/lunar_lander/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/evaluate_pbo_deep_%a.out -p amd,amd2 launch_job/lunar_lander/evaluate_pbo_deep.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -a deep $CONV)
fi


if [[ $IDQN = true ]]
then
    # IDQN
    echo "launch train idqn"
    submission_train_idqn=$(sbatch -J train_idqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=750Mc --time=5:30:00 --output=out/lunar_lander/$EXPERIMENT_NAME/train_idqn_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/train_idqn_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2 launch_job/lunar_lander/train_idqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION -g)

    IFS=" " read -ra split_submission_train_idqn <<< $submission_train_idqn
    submission_id_train_idqn=${split_submission_train_idqn[-1]}

    echo "launch evaluate idqn"
    submission_evaluate_idqn=$(sbatch -J evaluate_idqn --dependency=afterok:$submission_id_train_idqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=4000Mc --time=15:00 --output=out/lunar_lander/$EXPERIMENT_NAME/evaluate_idqn_%a.out --error=error/lunar_lander/$EXPERIMENT_NAME/evaluate_idqn_%a.out -p amd,amd2 launch_job/lunar_lander/evaluate_idqn.sh -e $EXPERIMENT_NAME -b $MAX_BELLMAN_ITERATION)
fi 
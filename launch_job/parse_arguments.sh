#!/bin/bash

function parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e | --experiment_name)
                EXPERIMENT_NAME=$2
                shift
                shift
                ;;
            -fs | --first_seed)
                FIRST_SEED=$2
                shift
                shift
                ;;
            -ls | --last_seed)
                LAST_SEED=$2
                shift
                shift
                ;;
            -b | --max_bellman_iterations)
                MAX_BELLMAN_ITERATION=$2
                shift
                shift
                ;;
            -sfqi | --skip_fqi)
                FQI=false
                shift
                ;;
            -slspi | --skip_lspi)
                LSPI=false
                shift
                ;;
            -sdqn | --skip_dqn)
                DQN=false
                shift
                ;;
            -spbo_linear | --skip_pbo_linear)
                PBO_linear=false
                shift
                ;;
            -spbo_max_linear | --skip_pbo_max_linear)
                PBO_max_linear=false
                shift
                ;;
            -spbo_custom_linear | --skip_pbo_custom_linear)
                PBO_custom_linear=false
                shift
                ;;
            -spbo_deep | --skip_pbo_deep)
                PBO_deep=false
                shift
                ;;
            -spbo_optimal | --skip_pbo_optimal)
                PBO_optimal=false
                shift
                ;;
            -c | --conv)
                CONV="-c"
                shift
                ;;
            -g | --gpu)
                GPU=true
                shift
                ;;
            -?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                shift
                ;;
            ?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                ;;
        esac
    done

    if [[ $EXPERIMENT_NAME == "" ]]
    then
        echo "experiment name is missing, use -e" >&2
        exit
    elif ( [[ $FIRST_SEED != "" ]] && [[ $LAST_SEED = "" ]] ) || ( [[ $FIRST_SEED == "" ]] && [[ $LAST_SEED != "" ]] )
    then
        echo "you need to specify -fs and -ls, not only one" >&2
        exit
    elif [[ $MAX_BELLMAN_ITERATION == "" ]]
    then
        echo "max_bellman_iterations is missing, use -b" >&2
        exit
    fi
    if [[ $FQI == "" ]]
    then
        FQI=true
    fi
    if [[ $LSPI == "" ]]
    then
        LSPI=true
    fi
    if [[ $DQN == "" ]]
    then
        DQN=true
    fi
    if [[ $PBO_linear == "" ]]
    then
        PBO_linear=true
    fi
    if [[ $PBO_max_linear == "" ]]
    then
        PBO_max_linear=true
    fi
    if [[ $PBO_custom_linear == "" ]]
    then
        PBO_custom_linear=true
    fi
    if [[ $PBO_deep == "" ]]
    then
        PBO_deep=true
    fi
    if [[ $PBO_optimal == "" ]]
    then
        PBO_optimal=true
    fi
    if [[ $GPU == "" ]]
    then
        GPU=false
    fi
}
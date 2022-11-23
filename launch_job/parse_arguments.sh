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
            -s | --seed)
                SEED=$2
                shift
                shift
                ;;
            -b | --max_bellman_iterations)
                MAX_BELLMAN_ITERATION=$2
                shift
                shift
                ;;
            -a | --architecture)
                shift
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
    fi

    if [[ $SEED == "" ]] && [[ $FIRST_SEED == "" ]] && [[ $LAST_SEED == "" ]]
    then
        echo "seed information is missing, use -s or (-fs and -ls)" >&2
        exit
    elif [[ $SEED != "" ]] && ( [[ $FIRST_SEED != "" ]] || [[ $LAST_SEED != "" ]] )
    then
        echo "you need to choose between -s and (-fs and -ls)" >&2
        exit
    elif [[ $SEED == "" ]] && ( [[ $FIRST_SEED == "" ]] || [[ $LAST_SEED == "" ]] )
    then
        echo "you need to specify -fs and -ls" >&2
        exit
    elif [[ $MAX_BELLMAN_ITERATION == "" ]]
    then
        echo "max_bellman_iterations is missing, use -b" >&2
        exit
    fi
}
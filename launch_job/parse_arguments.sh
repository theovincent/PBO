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
            -b | --bellman_iterations_scope)
                BELLMAN_ITERATIONS_SCOPE=$2
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
    elif ( [[ $FIRST_SEED != "" ]] && [[ $LAST_SEED = "" ]] ) || ( [[ $FIRST_SEED == "" ]] && [[ $LAST_SEED != "" ]] )
    then
        echo "you need to specify the first and last seed, use -fs and -ls" >&2
        exit
    fi
}
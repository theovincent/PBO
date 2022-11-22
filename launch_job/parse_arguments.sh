#!/bin/bash

function parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
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
            -c | --collect_sample)
                COLLECT_SAMPLE=true
                shift
                ;;
            -?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                shift
                ;;
        esac
    done

    if [[ $SEED == "" ]]
    then
        echo "seed is missing, use -s" >&2
        exit
    elif [[ $MAX_BELLMAN_ITERATION == "" ]]
    then
        echo "max_bellman_iterations is missing, use -b" >&2
        exit
    fi

    if [[ $COLLECT_SAMPLE == "" ]]
    then
        COLLECT_SAMPLE=false
    fi
}
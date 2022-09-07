function parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -bi | --n_bellman_iterations)
                N_BI=$2
                shift
                shift
                ;;
            -s | --n_seeds)
                N_SEEDS=$2
                shift
                shift
                ;;
            -?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                ;;
        esac
    done

    if [[ $N_BI == "" ]]
    then
        echo "N_BI is missing, use -bi" >&2
        exit
    elif [[ $N_SEEDS == "" ]]
    then
        echo "N_SEEDS is missing, use -s" >&2
        exit
    fi
}
#!/bin/bash

source env_cpu/bin/activate 

car_on_hill_pbo_evaluate -s $SEED -b $MAX_BELLMAN_ITERATION -a linear
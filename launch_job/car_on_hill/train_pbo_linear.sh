#!/bin/bash

source env_gpu/bin/activate 

car_on_hill_pbo -s $SEED -b $MAX_BELLMAN_ITERATION -a linear
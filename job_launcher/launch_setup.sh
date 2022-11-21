#!/bin/bash

submission_run=$(sbatch -J setup --mem-per-cpu=2000Mc --time 3 -o out/setup.out fit_running/stage/run.sh)
echo $submission_run
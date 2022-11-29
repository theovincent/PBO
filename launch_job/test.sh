#!/bin/bash

nvidia-smi

source env_gpu/bin/activate

python test.py

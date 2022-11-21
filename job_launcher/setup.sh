#!/bin/bash

python3 -m venv env_gpu
source env_gpu/bin/activate
pip install -e .[gpu]
pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
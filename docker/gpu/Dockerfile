FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN mkdir /workspace
WORKDIR /workspace

COPY . .

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip install --upgrade pip

RUN pip install -e . && pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
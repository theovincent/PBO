FROM python:3.8.13-buster

RUN mkdir /workspace
WORKDIR /workspace

RUN pip install --upgrade pip

COPY . .

RUN pip install -e .
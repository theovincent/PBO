In the folder where the code is, build the image and run the container in iterative mode.

For CPU usage:
```Bash
docker build -t pbo_image -f docker/cpu/Dockerfile .
docker run -it --mount type=bind,src=`pwd`/experiments/,dst=/workspace/experiments/ pbo_image bash
```

For GPU usage:
```Bash
docker build -t pbo_image -f docker/gpu/Dockerfile .
docker run -it --gpus all --mount type=bind,src=`pwd`/experiments/,dst=/workspace/experiments/ pbo_image bash
```
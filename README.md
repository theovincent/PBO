# Projected Bellman Operator (PBO)

## User installation
### Without Docker, with Python 3.8 or 3.9 installed
In the folder where the code is, create a Python virtual environment, activate it and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install -e .
```

### With Docker
Please see the [README](./docker/README.md) file made for that.

## Run the experiments
All the experiments can be ran the same way by simply replacing the name of the environment, here is an example for LQR.

The following command line runs the training and the evaluation of all the algorithms, one after the other:
```Bash
launch_job/lqr/launch_local.sh --experiment_name test --max_bellman_iterations 3 --first_seed 1 --last_seed 1
```
The expected time to finish the runs is 1 minute.

Once all the trainings are done, you can generate the figures shown in the paper by running the jupyter notebook file located at *experiments/lqr/plots.ipynb*. In the first cell of the notebook, please make sure to change the *experiment_name*, the *max_bellman_iterations* and the *seeds* according to the training that you have ran. You can also have a look at the loss of the training thought the jupyter notebook under *experiments/lqr/plots_loss.ipynb*.

## Run the tests
Run all tests with
```Bash
pytest
```
The code should take around 1 minutes to run.


## Using a GPU
In the folder where the code is, create a Python virtual environment, activate it and install the package and its dependencies in editable mode:
```bash
python3 -m venv env_gpu
source env_gpu/bin/activate
pip install -e .
pip install -U jax[cuda11_cudnn82]==0.3.22 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
(Taken from https://github.com/google/jax/discussions/10323)


## Using a cluster
Download miniconda on the server host to get Python 3.8:
```Bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Install cuda packages with:
```Bash
conda install -c conda-forge cudatoolkit-dev
```
do not forget to set the environment variable *LD_LIBRARY_PATH* correctly.
Finally, upgrade pip and install virtualenv
```Bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
Now you can go back to the [user installation](#user-installation) guidelines.
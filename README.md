# Projected Bellman Operator (PBO)

## User installation
### Without Docker, with Python 3.8 or 3.9 installed
In the folder where the code is, create a Python virtual environment, activate it and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install -e .[cpu]
```

### With Docker
In the folder where the code is, build the image and run the container in iterative mode:
```bash
docker build -t pbo_image .
docker run -it --mount type=bind,src=`pwd`/figure_specific,dst=/workspace/figure_specific pbo_image bash
```

## Run the experiments
For an `environment` and an `algorithm`, a jupyter notebook running the associated the experience can be found at _figure_specific/[environment]/[algorithm].ipynb_.

For example, the jupyter notebook _figure_specific/chain_walk/PBO_linear.ipynb_ trains a linear PBO on the Chain-Walk environment.

To generate the plots with $N$ seeds with $K$ Bellman iterations, you first need to generate the data by running `./figure_specific/[environment]/run_seeds.sh -n_seeds N -n_bellman_iteration K`
and then run the jupyter notebook _figure_specific/[environment]/plots.ipynb_.

### Replicate figures
Figure 4a with one seed, run
```Bash
./figure_specific/chain_walk/run_seeds.sh --n_seeds 1 --n_bellman_iterations 5
jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/plots.ipynb
```
You will find Figure 4a at _figure_specific/chain_walk/figures/distance_to_optimal_V_5.pdf_. The code should take around 5 minutes to run.

Figure 4b with one seed, run
```Bash
./figure_specific/lqr/run_seeds.sh --n_seeds 1 --n_bellman_iterations 2
jupyter nbconvert --to notebook --inplace --execute figure_specific/lqr/plots.ipynb
```
You will find Figure 4b at _figure_specific/lqr/figures/distance_to_optimal_Pi_2.pdf_. The code should take around 2 minutes to run.

Figure 5a with one seed, run
```Bash
./figure_specific/car_on_hill/run_seeds.sh --n_seeds 1 --n_bellman_iterations 9
jupyter nbconvert --to notebook --inplace --execute figure_specific/car_on_hill/samples.ipynb
jupyter nbconvert --to notebook --inplace --execute figure_specific/car_on_hill/plots.ipynb
```
You will find Figure 5a at _figure_specific/car_on_hill/figures/distance_to_optimal_V_9.pdf_. The code should take around 30 minutes to run.

Figure 5b with one seed, run
```Bash
./figure_specific/bicycle/run_seeds.sh --n_seeds 1 --n_bellman_iterations 8
jupyter nbconvert --to notebook --inplace --execute figure_specific/bicycle/plots.ipynb
```
You will find Figure 5a at _figure_specific/bicycle/figures/seconds_8.pdf_. The code should take around 45 minutes to run.

If any problem is encountered, make sure your files match the [file organization](#file-organization) and that the parameters _figure_specific/[environment]/plots.ipynb_ are matching the data that has been computed so far.

## Run the tests
Run all tests with
```Bash
pytest
```
The code should take around 1 minutes to run.

## File organization
```
📦PBO
 ┣ 📂env  # environment files
 ┣ 📂figure_specific  # files to run the experiments
 ┃ ┣ 📂bicycle
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┣ 📂data
 ┃ ┃ ┃ ┃ ┣ 📂FQI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy  # data generated by running the experiments
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear_max_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┗ 📂optimal
 ┃ ┃ ┃ ┃   ┗ 📜...npy
 ┃ ┃ ┃ ┗ 📜...pdf  # plots of the experiments generated from plots.ipynb
 ┃ ┃ ┣ 📜FQI.ipynb
 ┃ ┃ ┣ 📜PBO_linear_max_linear.ipynb
 ┃ ┃ ┣ 📜PBO_linear.ipynb
 ┃ ┃ ┣ 📜parameters.json
 ┃ ┃ ┣ 📜plots.ipynb
 ┃ ┃ ┗ 📜run_seeds.sh
 ┃ ┣ 📂car_on_hill
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┣ 📂data
 ┃ ┃ ┃ ┃ ┣ 📂FQI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear_max_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┗ 📂optimal
 ┃ ┃ ┃ ┃   ┗ 📜...npy
 ┃ ┃ ┃ ┗ 📜...pdf
 ┃ ┃ ┣ 📜FQI.ipynb
 ┃ ┃ ┣ 📜PBO_linear_max_linear.ipynb
 ┃ ┃ ┣ 📜PBO_linear.ipynb
 ┃ ┃ ┣ 📜optimal.py
 ┃ ┃ ┣ 📜parameters.json
 ┃ ┃ ┣ 📜plots.ipynb
 ┃ ┃ ┗ 📜run_seeds.sh
 ┃ ┣ 📂chain_walk
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┣ 📂data
 ┃ ┃ ┃ ┃ ┣ 📂FQI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂LSPI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_max_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┗ 📂optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┗ 📜...pdf
 ┃ ┃ ┣ 📜FQI.ipynb
 ┃ ┃ ┣ 📜LSPI.ipynb
 ┃ ┃ ┣ 📜PBO_linear.ipynb
 ┃ ┃ ┣ 📜PBO_max_linear.ipynb
 ┃ ┃ ┣ 📜PBO_optimal.ipynb
 ┃ ┃ ┣ 📜optimal.ipynb
 ┃ ┃ ┣ 📜parameters.json
 ┃ ┃ ┣ 📜plots.ipynb
 ┃ ┃ ┗ 📜run_seeds.sh
 ┃ ┗ 📂lqr
 ┃   ┣ 📂figures
 ┃   ┃ ┣ 📂data
 ┃   ┃ ┃ ┣ 📂FQI
 ┃   ┃ ┃ ┃ ┗ 📜...npy
 ┃   ┃ ┃ ┣ 📂LSPI
 ┃   ┃ ┃ ┃ ┗ 📜...npy
 ┃   ┃ ┃ ┣ 📂PBO_custom_linear
 ┃   ┃ ┃ ┃ ┗ 📜...npy
 ┃   ┃ ┃ ┣ 📂PBO_linear
 ┃   ┃ ┃ ┃ ┗ 📜...npy
 ┃   ┃ ┃ ┣ 📂PBO_optimal
 ┃   ┃ ┃ ┃ ┗ 📜...npy
 ┃   ┃ ┃ ┗ 📂optimal
 ┃   ┃ ┃ ┃ ┗ 📜...npy
 ┃   ┃ ┗ 📜...pdf
 ┃   ┣ 📜FQI.ipynb
 ┃   ┣ 📜LSPI.ipynb
 ┃   ┣ 📜PBO_custom_linear.ipynb
 ┃   ┣ 📜PBO_linear.ipynb
 ┃   ┣ 📜PBO_optimal.ipynb
 ┃   ┣ 📜optimal.ipynb
 ┃   ┣ 📜parameters.json
 ┃   ┣ 📜plots.ipynb
 ┃   ┗ 📜run_seeds.sh
 ┣ 📂test  # tests for the environments and the networks
 ┗ 📂pbo  # main code
```
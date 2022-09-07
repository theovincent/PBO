# Projected Bellman Operator (PBO)

## User installation
### Without Docker
In the folder where the code is, create a Python virtual environment, activate it and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install -e .[cpu]
```

### With Docker
In the folder where the code is, build the image and run the container in iterative mode:
```bash
docker build -t PBO
docker run -it PBO bash
```

## Run the experiments
For an `environment` and an `algorithm`, a jupyter notebook running the associated the experience can be found at _figure_specific/[environment]/[algorithm].ipynb_.

For example, the jupyter notebook _figure_specific/chain_walk/PBO_linear.ipynb_ trains a linear PBO on the Chain-Walk environment.

To generate the plots with $N$ seeds, you first need to generate the data by running `./figure_specific/[environment]/run_seeds.sh N`
and then running the jupyter notebook _figure_specific/[environment]/plots.ipynb_.

For example, the generate Figure 4a with only 2 seeds, you can run
```Bash
./figure_specific/chain_walk/run_seeds.sh 2
jupyter nbconvert --to notebook --inplace --execute figure_specific/chain_walk/plots.ipynb
```
Please make sure that the parameter `max_bellman_iterations` in _figure_specific/chain_walk/plots.ipynb_ matches with the parameter in _figure_specific/chain_walk/parameters.json_ and that `n_seeds = 2` in _figure_specific/chain_walk/plots.ipynb_.

If any problem is encountered, make sure your files match the [file organization](#file-organization).




## File organization
```
📦PBO
 ┣ 📂env  # environment files
 ┣ 📂figure_specific  # files to run the experiments
 ┃ ┣ 📂car_on_hill
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┣ 📂data
 ┃ ┃ ┃ ┃ ┣ 📂FQI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy  # data generated by running the experiments
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_max_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┗ 📂optimal
 ┃ ┃ ┃ ┃   ┗ 📜...npy
 ┃ ┃ ┃ ┗ 📜...pdf  # plots of the experiments generated from plots.ipynb
 ┃ ┃ ┣ 📜FQI.ipynb
 ┃ ┃ ┣ 📜PBO_deep.ipynb
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
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┣ 📂data
 ┃ ┃ ┃ ┃ ┣ 📂FQI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂LSPI
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_custom_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_linear
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┣ 📂PBO_optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┃ ┗ 📂optimal
 ┃ ┃ ┃ ┃ ┃ ┗ 📜...npy
 ┃ ┃ ┃ ┗ 📜...pdf
 ┃ ┃ ┣ 📜FQI.ipynb
 ┃ ┃ ┣ 📜LSPI.ipynb
 ┃ ┃ ┣ 📜PBO_custom_linear.ipynb
 ┃ ┃ ┣ 📜PBO_linear.ipynb
 ┃ ┃ ┣ 📜PBO_optimal.ipynb
 ┃ ┃ ┣ 📜optimal.ipynb
 ┃ ┃ ┣ 📜parameters.json
 ┃ ┃ ┣ 📜plots.ipynb
 ┃ ┃ ┗ 📜run_seeds.sh
 ┗ 📂pbo  # main code
```

You might need to run this
pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
from https://github.com/google/jax/discussions/10323

## TO DO 
installation (python/docker)
use
run test
organization


# Usage

The instructions for running MLCSHE search as well as the baseline search methods are provided in this section. They all run on the pylot case study. More details on the case study, the encodings, accessing and reproducing the results is provided in [pylot/README.md](pylot/README.md).

Before running any of the search methods, in a separate terminal tab, run the following command:

```bash
watch nvidia-smi
```

The output of the command should look similar to the following:

```bash
Wed Jan 25 12:07:31 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   46C    P8     2W /  N/A |     10MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1306      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A      2065      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

Keep this tab running in the background. The output of the command will be used to monitor the GPU memory usage during the search. For each simulation, 3 processes would run the nvidia-smi output, 1 for Carla and 2 for Pylot. In case of 2 parallel simulations, which is the default setting, 2 sets of 3 processes would be running. If you want to run more simulations in parallel, you need to make sure that your machine has enough resources (especially RAM and GPU memory) to run the jobs in parallel. More info on resources is provided in the [installation guide](INSTALL.md).

*NOTE*: If you do not see a similar command, you need to make sure proper drivers are installed. You can find more information on installing the drivers in the [installation guide](INSTALL.md).

## Running MLCSHE

To run MLCSHE, you can use the following command:

```bash
python run_mlcshe.py
```

*NOTE 1*: the parameters of the search can be updated in `~/pylot/search_config.py`. The parameters of the simulation can be updated in `~/simulation_config.py`.

*NOTE 2*: the simulations are run in `2` parallel jobs. If you want to run more jobs, you can update the `num_jobs` variable in `~/simulation_config.py`. However, you need to make sure that your machine has enough resources (especially RAM and GPU memory) to run the jobs in parallel.

## Running Baseline Search Methods

- To run Random Search, you can use the following command:

```bash
python run_random_search.py <SIM_BUDGET>
```

where `<SIM_BUDGET>` is the number of simulations to run.

- To run Genetic Algorithm Search, you can use the following command:

```bash
python run_ga_search.py <SIM_BUDGET> <POP_SIZE> <MAX_NUM_GEN>
```

where `<SIM_BUDGET>`, `<POP_SIZE>` and `<MAX_NUM_GEN>` are the number of simulations to run, the population size and the maximum number of generations, respectively.
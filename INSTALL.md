# Installation Guide

This document provides that installation guidelines on an ubuntu 20.04 LTS machine.

## Prerequisites

+ Python 3.7 or newer.
+ Docker 20.10 or newer.
+ NVIDIA driver 525.60 or newer.

## AWS EC2 Instance

If your ubuntu machine is running as an AWS EC2 instance, please follow these steps first to setup your machine.

+ Choose a *g4dn* image, with a minimum size of *xlarge* (for parallel execution of 2 simulations, a minimum size of *2xlarge* is required).
+ Choose a *NVidia GPU-optimized AMI*.

## Steps

1. Install nvidia-docker2 using the following commands, or using the guide provided [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker):

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Pull the modified Pylot Docker image from Docker Hub:

```bash
docker pull sepshr/pylot:2.1
```

<!-- *NOTE:* In the code, it is assumed that the name of the docker container is `pylot`. If you are using another name, please ensure that you update the `container_name` variable inside `/MLCSHE/simulation_config.py`. -->

<!-- 3.Next, setup SSH connection for the container. First, add your public ssh key to the `~/.ssh/authorized_keys` in the container:

```bash
nvidia-docker cp ~/.ssh/id_rsa.pub pylot:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo chown erdos /home/erdos/.ssh/authorized_keys
``` -->

(*Optional*) To test Carla and Pylot, first run Carla in the container using the following command:

```bash
nvidia-docker exec -i -t pylot /home/erdos/workspace/pylot/scripts/run_simulator.sh
```

(*Optional*) Then, in another terminal window, run Pylot in the container using the following command:

```bash
nvidia-docker exec -i -t pylot /bin/bash
cd workspace/pylot/
python3 pylot.py --flagfile=configs/detection.conf
```

3. Clone the MLCSHE project on your machine. For now, use the *Dev-Sepehr* branch.

```bash
git clone -b Dev-Sepehr https://github.com/SepShr/MLCSHE.git
```

4. Create a virtual environment and install the `requirements.txt` file.

```bash
python3 -m venv ./venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5a. To run MLCSHE, you can use the following command:

```bash
python run_mlcshe.py
```

*NOTE 1*: the parameters of the search can be updated in `search_config.py`. The parameters of the simulation can be updated in `simulation_config.py`.

*NOTE 2*: the simulations are run in `2` parallel jobs. If you want to run more jobs, you can update the `num_jobs` variable in `simulation_config.py`. However, you need to make sure that your machine has enough resources (especially RAM and GPU memory) to run the jobs in parallel.

5b. To run Random Search, you can use the following command:

```bash
python run_random_search.py <SIM_BUDGET>
```

where `<SIM_BUDGET>` is the number of simulations to run.

5c. To run Genetic Algorithm Search, you can use the following command:

```bash
python run_ga_search.py <SIM_BUDGET> <POP_SIZE> <MAX_NUM_GEN>
```

where `<SIM_BUDGET>`, `<POP_SIZE>` and `<MAX_NUM_GEN>` are the number of simulations to run, the population size and the maximum number of generations, respectively.

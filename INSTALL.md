# Installation Guide

This document provides that installation guidelines on an ubuntu 20.04 LTS machine.

## Prerequisites

+ Python 3.7 or newer.
+ Docker 20.10 or newer.
+ NVIDIA driver 525.60 or newer.
+ Free disk space of 150 GB or more.
+ 32 GB of RAM or more.

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

3. If you have not downloaded the code already (e.g., from FigShare), you can clone the MLCSHE project on your machine. For now, use the *Dev-Sepehr* branch. 

```bash
git clone -b Dev-Sepehr https://github.com/SepShr/MLCSHE.git
```

4. Navigate to the project's directory (in case of cloning the project, the `MLCSHE` directory) and create a virtual environment and install the `requirements.txt` file.

```bash
cd MLCSHE
python3 -m venv ./venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

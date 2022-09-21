# Installation Guide

This document provides that installation guidelines on an ubuntu 20.04 LTS machine.

## Prerequisites

+ Python 3.7 or newer.

## AWS EC2 Instance

If your ubuntu machine is running as an AWS EC2 instance, please follow these steps first to setup your machine.

+ Choose a *g4dn* image, with a minimum size of *xlarge*.
+ Choose a *NVidia GPU-optimized AMI*.

## Steps

1.Install nvidia-docker2 using the following commands, or using the guide provided [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker):

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2.Pull the modified Pylot Docker image from Docker Hub:

```bash
docker pull sepshr/pylot:2.1
```

<!-- *NOTE:* In the code, it is assumed that the name of the docker container is `pylot`. If you are using another name, please ensure that you update the `container_name` variable inside `/MLCSHE/simulation_config.py`. -->

<!-- 3.Next, setup SSH connection for the container. First, add your public ssh key to the `~/.ssh/authorized_keys` in the container:

```bash
nvidia-docker cp ~/.ssh/id_rsa.pub pylot:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo chown erdos /home/erdos/.ssh/authorized_keys
``` -->

3.Clone the MLCSHE project on your machine. For now, use the *Dev-Sepehr* branch.

```bash
git clone -b Dev-Sepehr https://github.com/SepShr/MLCSHE.git
```

4.Create a virtual environment and install the `requirements.txt` file.

```bash
python3 -m venv ./venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5.To run the simulator, you can use the following command:

```bash
python run_iccea.py
```

# Installation Guide
This document provides that installation guidelines on an ubuntu 20.04 LTS machine.

## AWS EC2 Instance
If your ubuntu machine is running as an AWS EC2 instance, please follow these steps first to setup your machine.

g4dn.xlarge image
NVidia machine learning AMI
Install nvidia-docker2

## Steps
1. Pull the modified Pylot Docker image from Docker Hub:
```bash
docker pull sepshr/pylot:1.0
nvidia-docker run -itd --name pylot -p 20022:22 erdosproject/pylot /bin/bash
```

*NOTE:* In the code, it is assumed that the name of the docker container is `pylot`. If you are using another name, please ensure that you update the `container_name` variable inside `/MLCSHE/simulation_config.py`. Additionally, you have to change the name of container in the 2 script files, namely `run_pylot.sh` and `copy_pylot_finished_file.sh`.

2. Next, setup SSH connection for the container. First, add your public ssh key to the `~/.ssh/authorized_keys` in the container:
```bash
nvidia-docker cp ~/.ssh/id_rsa.pub pylot:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo chown erdos /home/erdos/.ssh/authorized_keys
```

3. Clone the MLCSHE project on your machine. For now, use the *Dev-Sepehr* branch.

4. Make sure that you create a `results` folder in `MLCSHE`'s working directory with the `mkdir results` command.

5. Create a virtual environment and install the `requirements.txt` file.

6. To run the simulator, you can use the following command:
```bash
python3 run_iccea.py
```
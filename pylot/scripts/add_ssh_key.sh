#!/bin/bash
nvidia-docker cp ~/.ssh/id_rsa.pub $1:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t $1 sudo chown erdos /home/erdos/.ssh/authorized_keys
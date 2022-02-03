#!/bin/bash
nvidia-docker exec -i -t pylot_main sudo service ssh start 
ssh -p 20022 -X erdos@localhost 'cd /home/erdos/workspace/pylot/;export PYTHONPATH="$PYTHONPATH:$PYLOT_HOME/dependencies/lanenet/";python3 pylot.py --flagfile=configs/mlcshe/mlcshe_config.conf'

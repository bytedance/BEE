#!/bin/bash

cd /opt/tiger/E2E
echo $PYTHONPATH
echo $ARNOLD_ROLE, $ARNOLD_ID, $ARNOLD_SERVER_HOSTS, $ARNOLD_WORKER_HOSTS, $ARNOLD_OUTPUT
python3 -u /opt/tiger/E2E/Train/quant_model.py $@
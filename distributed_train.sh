#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --master_port 29575 --nproc_per_node=$NUM_PROC train.py "$@"


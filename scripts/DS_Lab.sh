#!/bin/bash

module load new gcc/4.8.2 python/3.7.1

today=`date +%Y-%m-%d_%H-%M-%S`

python scripts/train.py > Output/output_${today}.txt

cat ../Wandb_Auth.txt
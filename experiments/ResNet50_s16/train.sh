#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="log/rfcn_end2end_ResNet50_s16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ../../tools/train_net.py --gpu 13 \
  --solver ./solver_ohem.prototxt \
  --weights ./snapshot/resnet50_rfcn_ohem_s16_iter_50000.caffemodel \
  --imdb tt100k_train \
  --iters 40000 \
  --cfg ./config.yml \



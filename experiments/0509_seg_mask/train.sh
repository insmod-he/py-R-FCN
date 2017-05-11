#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="log/rfcn_end2end_ResNet50_s16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python -m pdb ../../tools/train_net.py --gpu 15 \
  --solver ./solver_ohem.prototxt \
  --weights /data2/HongliangHe/work2017/TrafficSign/py-R-FCN/data/imagenet_models/ResNet-50-model.caffemodel \
  --imdb tt100k_val32 \
  --iters 40000 \
  --cfg ./config.yml 


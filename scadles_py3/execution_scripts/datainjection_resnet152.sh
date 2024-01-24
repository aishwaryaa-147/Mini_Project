#!/bin/bash

# randomized data-injection for ResNet152 on IID/Non-IID CIFAR10.

cd ~/ScaDLES/scadles_py3/

model='resnet152'
dataset='noniid_cifar10'
basedir='/'
lr=0.1
weightdecay=1e-4
momentum=0.9
world_size=10
determinism=0
gamma=0.2
backend='mpi'
trainbsz=64
globalbsz=$((trainbsz * world_size))
streamfreqs=(270 218 300 246 232 284 284 236 274 228)
strmode='persistence'
alpha=0.25
beta=0.25

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  directory=$basedir's'$rank'/'
  streamfrq=${streamfreqs[$(($rank-1))]}
  echo 'launch training for rank '$rank' and procrank '$procrank ' and stream_freq '$streamfrq
  python3 -m launch.datainjection_resnet152 --dir=$directory --model-name=$model --datatype=$dataset \
  --train-bsz=$trainbsz --gamma=$gamma --weight-decay=$weightdecay --lr=$lr --momentum=$momentum \
  --world-size=$world_size --backend=$backend --local-rank=$procrank --global-bsz=$globalbsz \
  --global-rank=$procrank --stream-freq=$streamfrq --stream-mode=$strmode --alpha=$alpha --beta=$beta &
done
#!/bin/bash

# randomized data-injection for VGG19 on IID/Non-IID CIFAR100.

cd ~/ScaDLES/scadles_py3/

model='vgg19'
dataset='noniid_cifar100'
basedir='/'
determinism=0
lr=0.01
weightdecay=5e-4
momentum=0.9
gamma=0.3
world_size=8
backend='mpi'
trainbsz=64
globalbsz=$((trainbsz * world_size))
streamfreqs=(92 22 62 42 48 36 18 38 84 60 108 40 58 16 44 78 92 28 30 52 84 20 16 38 66)
strmode='persistence'
alpha=0.25
beta=0.25

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  directory=$basedir's'$rank'/'
  streamfrq=${streamfreqs[$(($rank-1))]}
  echo 'launch training for rank '$rank' and procrank '$procrank ' and stream_freq '$streamfrq
  python3 -m launch.datainjection_vgg19 --dir=$directory --model-name=$model --datatype=$dataset \
  --train-bsz=$trainbsz --gamma=$gamma --weight-decay=$weightdecay --lr=$lr --momentum=$momentum \
  --world-size=$world_size --backend=$backend --local-rank=$procrank --global-bsz=$globalbsz \
  --global-rank=$procrank --stream-freq=$streamfrq --stream-mode=$strmode --alpha=$alpha --beta=$beta &
done
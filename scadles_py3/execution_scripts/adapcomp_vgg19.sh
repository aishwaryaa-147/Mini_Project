#!/bin/bash

# adaptive compression for VGG19 on IID/Non-IID CIFAR100.

cd ~/ScaDLES/scadles_py3/

basedir='/'
model='vgg19'
dataset='iid_cifar100'
determinism=0
lr=0.01
weightdecay=5e-4
momentum=0.9
gamma=0.3
world_size=25
backend='mpi'
trainbsz=64
globalbsz=$((trainbsz * world_size))
streamfreqs=(92 22 62 42 48 36 18 38 84 60 108 40 58 16 44 78 92 28 30 52 84 20 16 38 66)
strmode='persistence'
# compression factor 10x
compressratio=0.1
compressthreshold=0.3

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  directory=$basedir's'$rank'/'
  streamfrq=${streamfreqs[$(($rank-1))]}
  echo 'launch training for rank '$rank' and procrank '$procrank ' and stream_freq '$streamfrq
  python3 -m launch.adaptive_compression --dir=$directory --model-name=$model --datatype=$dataset \
  --train-bsz=$trainbsz --gamma=$gamma --weight-decay=$weightdecay --lr=$lr --momentum=$momentum \
  --world-size=$world_size --kafka-dir=$kafkadir --backend=$backend --local-rank=$procrank --global-bsz=$globalbsz \
  --global-rank=$procrank --stream-freq=$streamfrq --stream-mode=$strmode --compression-ratio=$compressratio \
  --delta-threshold=$compressthreshold &
done
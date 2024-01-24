#!/bin/bash

# adaptive compression with randomized data-injection for ResNet152 on IID/Non-IID CIFAR10.

cd ~/ScaDLES/scadles_py3/

basedir='/'
model='resnet152'
dataset='noniid_cifar10'
lr=0.1
weightdecay=1e-4
momentum=0.9
world_size=10
determinism=0
gamma=0.2
backend='mpi'
trainbsz=64
globalbsz=$((trainbsz * world_size))
# compression factor 10x
compressratio=0.1
compressthreshold=0.3
alpha=0.5
beta=0.5

# Uniform dist S1
#streamfreqs=(50 38 12 36 34 13 82 40)
# Uniform dist S2
#streamfreqs=(270 218 300 246 232 284 284 236 274 228)
# Normal dist S1
streamfreqs=(92 38 62 42 48 88 90 22 44 28)
streammode='persistence'

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  directory=$basedir's'$rank'/'
  streamfrq=${streamfreqs[$(($rank-1))]}
  echo 'launch training for rank '$rank' and procrank '$procrank ' and str_freq '$streamfrq
  python3 -m launch.adaptivecompression_datainjection --dir=$directory --model-name=$model --datatype=$dataset \
  --train-bsz=$streamfrq --gamma=$gamma --weight-decay=$weightdecay --lr=$lr --momentum=$momentum \
  --world-size=$world_size --kafka-dir=$kafkadir --backend=$backend --local-rank=$procrank --global-bsz=$globalbsz \
  --global-rank=$procrank --stream-freq=$streamfrq --stream-mode=$streammode --compression-ratio=$compressratio \
  --delta-threshold=$compressthreshold --alpha=$alpha --beta=$beta &
done
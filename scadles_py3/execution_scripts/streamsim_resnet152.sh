#!/bin/bash

# streaming simulation for ResNet152 on IID/Non-IID CIFAR10.

cd ~/ScaDLES/scadles_py3/

model='resnet152'
dataset='iid_cifar10'
stream='persistence'
basedir='/'
kafkadir='/'
backend='mpi'
lr=0.4
weightdecay=1e-4
momentum=0.9
world_size=10
maxtrainbsz=64
globalbsz=$((maxtrainbsz * world_size))
determinism=0
gamma=0.1

# approx. 64 images/sec data-rate
data_rates=('0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015')

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  directory=$basedir's'$rank'/'
  echo 'launch training for rank '$rank' and procrank '$procrank
  python3 -m launch.train_streamsimulate --dir=$directory --model-name=$model --datatype=$dataset \
  --max-train-bsz=$maxtrainbsz --stream-mode=$stream --weight-decay=$weightdecay --lr=$lr --momentum=$momentum \
  --world-size=$world_size --gamma=$gamma --kafka-dir=$kafkadir --local-rank=$procrank --global-bsz=$globalbsz \
  --global-rank=$procrank --backend=$backend &
done

sleep 5

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  rank_data_rate=${data_rates[$procrank]}
  directory=$basedir'pub'$procrank'/'
  python3 -m datastreaming.kafka_publisher $rank_data_rate $procrank $world_size $directory $dataset $determinism &
  echo 'launched publisher for rank '$procrank' with data rate '$rank_data_rate
done
#!/bin/bash

# streaming simulation for VGG19 on IID/Non-IID CIFAR100.

cd ~/ScaDLES/scadles_py3/

model='vgg19'
dataset='iid_cifar100'
stream='truncate'

basedir='/'
kafkadir='/'
backend='mpi'
dataset='iid_cifar100'
determinism=0
trainbsz=64
world_size=10
globalbsz=$((maxtrainbsz * world_size))
lr=0.01
weightdecay=5e-4
momentum=0.9
gamma=0.2

# approx. 64 images/sec data-rate
data_rates=('0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015' '0.015')

for rank in $(seq 1 $world_size)
do
  procrank=$(($rank-1))
  directory=$basedir's'$rank'/'
  echo 'launch training for rank '$rank' and procrank '$procrank
  python3 -m launch.train_streamsimulate --dir=$directory --model-name=$model --datatype=$dataset --max-train-bsz=$maxtrainbsz \
  --stream-mode=$stream --gamma=$gamma --weight-decay=$weightdecay --lr=$lr --momentum=$momentum \
  --world-size=$world_size --local-rank=$procrank --kafka-dir=$kafkadir --global-bsz=$globalbsz \
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
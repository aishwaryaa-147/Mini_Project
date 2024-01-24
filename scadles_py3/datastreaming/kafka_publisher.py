from kafka import KafkaProducer
import scadles_py3.datastreaming.iid_data as iid_data
import scadles_py3.datastreaming.noniid_data as noniid_data

import time
import sys
import os

import torch

class DataPublisher(object):

    def __init__(self, data_rate, rank, world_sz, kafka_host, kafka_port, datatype, train_dir, seed, determinism):
        # given in seconds. describes the rate of flow of data into a given rank. eg. 0.1 means 10 events/sec
        self.data_rate = data_rate
        self.kafka_host = kafka_host
        self.kafka_port = kafka_port
        self.publish_rank = rank
        self.producer = KafkaProducer(bootstrap_servers=self.kafka_host+':'+self.kafka_port)
        self.datatype = datatype

        if self.datatype == 'iid_cifar10':
            self.loader = iid_data.IID_CIFAR10(train_dir=train_dir, world_size=world_sz, t_id=rank, seed=seed,
                                          determinism=determinism)
        elif self.datatype == 'iid_cifar100':
            self.loader = iid_data.IID_CIFAR100(train_dir=train_dir, world_size=world_sz, t_id=rank, seed=seed,
                                           determinism=determinism)
        # for noniid data self.loader is a simple list
        elif self.datatype == 'noniid_cifar10':
            self.loader = noniid_data.nonIID_CIFAR10(train_dir=train_dir, t_id=rank, seed=seed, determinism=determinism)
        elif self.datatype == 'noniid_cifar100':
            self.loader = noniid_data.nonIID_CIFAR100(train_dir=train_dir, t_id=rank, seed=seed, determinism=determinism)

    def start_streaming(self):
        if self.datatype == 'noniid_cifar10' or self.datatype == 'noniid_cifar100':
            batches = torch.utils.data.DataLoader(self.loader, batch_size=1, shuffle=True)
            del self.loader

        else:
            batches = torch.utils.data.DataLoader(self.loader, batch_size=1)
            records = []
            for input, label in batches:
                record = input, label
                records.append(record)

            del batches, self.loader
            batches = torch.utils.data.DataLoader(records, batch_size=1, shuffle=True)

        while True:
            for input, label in batches:
                self.producer.send(topic='trainer-' + str(self.publish_rank), key=label.numpy().tobytes(),
                                   value=input.numpy().tobytes())
                time.sleep(self.data_rate)


if __name__ == '__main__':
    # e.g. values of data_stream_rate, rank, world_sz, dir, dataset, determine = 0.1, 0, 1, '/','iid_cifar10', False
    data_stream_rate, rank, world_sz, dir, dataset, determine = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), \
                                            str(sys.argv[4]), str(sys.argv[5]), bool(int(sys.argv[6]))
    if not os.path.exists(dir):
        os.mkdir(dir)

    pub_stream = DataPublisher(data_rate=data_stream_rate, rank=rank, world_sz=world_sz , kafka_host='localhost',
                               kafka_port='9092', datatype=dataset, train_dir=dir,
                               seed=1234, determinism=determine)

    pub_stream.start_streaming()
from kafka import KafkaConsumer
import time
import numpy as np
import sys
import logging
import torch

## Implements stream-persistence and stream-truncation policies

class Waiting_Persistence_Streamer(object):

    def __init__(self, rank, train_bsz, input_shape=[1, 3, 32, 32], output_shape=[1]):
        self.topic = 'trainer-' + str(rank)
        self.train_bsz = train_bsz
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.consumer = KafkaConsumer(self.topic)

    def waiting_batched_streams(self):
        flag = True
        record = None
        for data in self.consumer:
            if flag:
                begin = time.time()
                flag = False

            input, label = torch.from_numpy(np.frombuffer(data.value, dtype=np.float32)).reshape(self.input_shape), \
                           torch.from_numpy(np.frombuffer(data.key, dtype=np.int32))[0].reshape(self.output_shape)

            if record is not None:
                input = torch.concat([record[0], input], dim=0)
                label = torch.concat([record[1], label], dim=0)

            record = input, label
            batchsize = input.size(dim=0)

            if batchsize >= self.train_bsz:
                rec = record[0][0: self.train_bsz], record[1][0: self.train_bsz]
                rec = [rec]
                record = record[0][self.train_bsz:], record[1][self.train_bsz:]

                wait_time = time.time() - begin
                flag = True
                yield rec, wait_time


class Waiting_Truncated_Streamer(object):

    def __init__(self, rank, train_bsz, input_shape=[1, 3, 32, 32], output_shape=[1]):
        self.topic = 'trainer-' + str(rank)
        self.train_bsz = train_bsz
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.consumer = KafkaConsumer(self.topic)

    def waiting_batched_streams(self):
        flag = True
        record = None
        for data in self.consumer:
            if flag:
                begin = time.time()
                flag = False

            input, label = torch.from_numpy(np.frombuffer(data.value, dtype=np.float32)).reshape(self.input_shape), \
                           torch.from_numpy(np.frombuffer(data.key, dtype=np.int32))[0].reshape(self.output_shape)

            if record is not None:
                input = torch.concat([record[0], input], dim=0)
                label = torch.concat([record[1], label], dim=0)

            record = input, label
            batchsize = input.size(dim=0)

            if batchsize >= self.train_bsz:
                rec = record[0][0: self.train_bsz], record[1][0: self.train_bsz]
                rec = [rec]
                del record
                record = None
                wait_time = time.time() - begin
                yield rec, wait_time


class Image_Persistence_Streamer(object):

    def __init__(self, rank, max_train_bsz, out_dir, input_shape=[1, 3, 32, 32], output_shape=[1]):
        self.topic = 'trainer-' + str(rank)
        self.max_train_bsz = max_train_bsz
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.consumer = KafkaConsumer(self.topic)
        logging.basicConfig(filename=out_dir + 'subscriber-' + str(rank) + '.log', level=logging.INFO)

    def batchify_streams(self):
        flag = True
        record = None
        ctr= 0
        for data in self.consumer:
            if flag:
                begin = time.time()
                flag = False

            input, label = torch.from_numpy(np.frombuffer(data.value, dtype=np.float32)).reshape(self.input_shape), \
                           torch.from_numpy(np.frombuffer(data.key, dtype=np.int32))[0].reshape(self.output_shape)

            if record is not None:
                input = torch.concat([record[0], input], dim=0)
                label = torch.concat([record[1], label], dim=0)

            record = input, label
            batchsize = input.size(dim=0)

            end = time.time()
            if end - begin >= 1.0:
                ctr += 1
                if batchsize <= self.max_train_bsz:
                    rec = [record]
                    logging.info(f'subscriber logging counter {ctr} w {record[0].size(dim=0)} x {record[0].size(dim=1)} y '
                                 f'{record[0].size(dim=2)} z {record[0].size(dim=3)}')
                    logging.info(f'LOGGING_TO_FILE ctr {ctr} batch_size {batchsize}')
                    input, label, record = None, None, None

                # when there are more records in stream that what is allowed
                else:
                    rec = record[0][0 : self.max_train_bsz], record[1][0 : self.max_train_bsz]
                    rec = [rec]
                    record = record[0][self.max_train_bsz :], record[1][self.max_train_bsz :]
                    logging.info(f'Xsubscriber logging counter {ctr} w {record[0].size(dim=0)} x {record[0].size(dim=1)} y '
                                 f'{record[0].size(dim=2)} z {record[0].size(dim=3)}')
                    logging.info(f'LOGGING_TO_FILE AT_LARGE ctr {ctr} batch_size {batchsize}')

                flag = True
                yield rec


class Image_Truncated_Streamer(object):

    def __init__(self, rank, max_train_bsz, input_shape=[1, 3, 32, 32], output_shape=[1]):
        self.topic = 'trainer-' + str(rank)
        self.max_train_bsz = max_train_bsz
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.consumer = KafkaConsumer(self.topic)

    def batchify_streams(self):
        flag = True
        record = None
        for data in self.consumer:
            if flag:
                begin = time.time()
                flag = False

            input, label = torch.from_numpy(np.frombuffer(data.value, dtype=np.float32)).reshape(self.input_shape).to(torch.float32), \
                     torch.from_numpy(np.frombuffer(data.key, dtype=np.int32))[0].reshape(self.output_shape).to(torch.int32)

            if record is not None:
                input = torch.concat([record[0], input], dim=0)
                label = torch.concat([record[1], label], dim=0)

            record = input, label
            batchsize = input.size(dim=0)
            end = time.time()
            if end - begin >= 1.0:
                if batchsize <= self.max_train_bsz:
                    rec = [record]
                # when there are more records in stream that what is allowed
                else:
                    rec = record[0][0 : self.max_train_bsz], record[1][0 : self.max_train_bsz]
                    rec = [rec]
                    del record

                flag = True
                record = None
                yield rec


if __name__ == '__main__':
    rank = int(sys.argv[1])
    out_dir = str(rank)             # stores each worker log to directory corresponding to rank
    persist = Image_Persistence_Streamer(rank=rank, max_train_bsz=1000, out_dir=out_dir)
    for loader in persist.batchify_streams():
        for input, label in loader:
            logging.info(f'STREAMING RATE gives batch_size {input.size()}')
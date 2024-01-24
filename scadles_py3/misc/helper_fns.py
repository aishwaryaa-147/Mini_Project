import numpy as np
from _random import Random
import random
import os
import psutil
import time
from prettytable import PrettyTable

import torch
import nvidia_smi

import scadles_py3.datastreaming.iid_data as iid_data
import scadles_py3.datastreaming.noniid_data as noniid_data

def set_seed(seed, determinism=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = Random()
    rng.seed(seed)
    torch.use_deterministic_algorithms(determinism)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gradient_norm(grads):
    gnorm = 0.0
    for g in grads:
        gnorm += torch.norm(g.flatten())

    return gnorm

def countparameters_memorysize(model):
    table = PrettyTable(["Module", "Parameters"])
    total_params = 0
    ctr = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
        ctr += 1

    print(table)
    print(f"layers {ctr}")
    print(f"Total Trainable Params: {total_params}")
    total_size = (total_params * 4) / (1024 * 1024)
    print(f"Gradient memory footprint using FP32: {total_size} MB")

    return total_params, total_size


def process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/(1024*1024)


def get_device_memory(t_id):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(t_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem = info.used
    nvidia_smi.nvmlShutdown()
    return gpu_mem


def get_dir_size(path='/'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def dataloadersize(model_memory, records, batchsizes=[1,16,32,64,128,256,512,768,1024,1536,2048], sample_itr=100):
    batchsize_memory = {}
    for sample_bs in batchsizes:
        batchsize_memory[sample_bs] = model_memory
        #init_mem = process_memory()
        loaded_mem = []
        loader = torch.utils.data.DataLoader(records, batch_size=sample_bs, prefetch_factor=1, num_workers=1)
        ctr = 0
        for input,_ in loader:
            ctr += 1
            loaded_mem.append(process_memory())
            time.sleep(0.1)
            if ctr == sample_itr:
                break

        # find memory consumption by process
        after_memory = np.mean(loaded_mem)
        dataloader_memory = after_memory
        print(f'bs {sample_bs} traced_memory {dataloader_memory} MB')
        batchsize_memory[sample_bs] += batchsize_memory[sample_bs] #+ dataloader_memory

        del loader

# computes the system memory consumed by the current process
def memoryconsumption(model_memory, dataset_name, train_dir, world_size, t_id, seed, determinism):
    if dataset_name == 'iid_cifar10':
        loader = iid_data.IID_CIFAR10(train_dir, world_size, t_id, seed, determinism)
    elif dataset_name == 'iid_cifar100':
        loader = iid_data.IID_CIFAR100(train_dir, world_size, t_id, seed, determinism)
    elif dataset_name == 'noniid_cifar10':
        loader = noniid_data.nonIID_CIFAR10(train_dir, t_id, seed, determinism)
    elif dataset_name == 'noniid_cifar100':
        loader = noniid_data.nonIID_CIFAR10(train_dir, t_id, seed, determinism)

    # if 'noniid' in dataset_name and isinstance(loader, list):
    dataloadersize(model_memory, loader)
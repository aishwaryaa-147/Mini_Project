import random

import torch
import torchvision
import torchvision.transforms as transforms

import scadles_py3.misc.helper_fns as misc

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartition(object):
    def __init__(self, data, world_size):
        self.data = data
        self.partitions = []
        # partition data equally among the trainers
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        partitions = [1 / (world_size) for _ in range(0, world_size)]
        print(f"training partitions {partitions}")

        for part in partitions:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0:part_len])
            print(f'for partition {part} length is {part_len}')
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def load_CIFAR10Test(log_dir, test_bsz, seed, determinism):
    misc.set_seed(seed, determinism)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=False,
                                           download=True, transform=transform)

    # shuffle set to False to test same order of samples across different configurations
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)
    return testloader

def load_CIFAR100Test(log_dir, test_bsz, seed, determinism):
    misc.set_seed(seed, determinism)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR100(root=log_dir, train=False,
                                            download=True, transform=transform)

    # shuffle set to False to test same order of samples across different configurations
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)
    return testloader
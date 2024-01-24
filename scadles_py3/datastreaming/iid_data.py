import torch
import torchvision
import torchvision.transforms as transforms

from scadles_py3.datastreaming.data_partitioner import DataPartition
import scadles_py3.misc.helper_fns as misc


class IID_CIFAR10(torch.utils.data.IterableDataset):

    def __init__(self, train_dir, world_size, t_id, seed, determinism):
        misc.set_seed(seed, determinism)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4), normalize])
        train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)

        self.traindata_iterable = DataPartition(train_data, world_size)
        self.traindata_iterable = self.traindata_iterable.use(t_id)

    def __iter__(self):
        yield from iter(self.traindata_iterable)

    def get_len(self):
        return len(self.traindata_iterable)


class IID_CIFAR100(torch.utils.data.IterableDataset):

    def __init__(self, train_dir, world_size, t_id, seed, determinism):
        misc.set_seed(seed, determinism)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4), normalize])
        train_data = torchvision.datasets.CIFAR100(root=train_dir, train=True, download=True, transform=transform)

        self.traindata_iterable = DataPartition(train_data, world_size)
        self.traindata_iterable = self.traindata_iterable.use(t_id)

    def __iter__(self):
        yield from iter(self.traindata_iterable)

    def get_len(self):
        return len(self.traindata_iterable)


def bulkloader_iidcifar10(train_dir, train_bsz, world_size, t_id, seed, determinism):
    misc.set_seed(seed, determinism)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)
    partition = DataPartition(train_data, world_size)
    partition = partition.use(t_id)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g,
                                              num_workers=1)
    return trainloader


def bulkloader_iidcifar100(train_dir, train_bsz, world_size, t_id, seed, determinism):
    misc.set_seed(seed, determinism)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR100(root=train_dir, train=True, download=True, transform=transform)
    partition = DataPartition(train_data, world_size)
    partition = partition.use(t_id)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g,
                                              num_workers=1)
    return trainloader
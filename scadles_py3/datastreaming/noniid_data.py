import logging

import torchvision
import torch
import torchvision.transforms as transforms

import scadles_py3.misc.helper_fns as misc

def nonIID_CIFAR10(train_dir, t_id, seed, determinism):
    label_val = t_id + 1
    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])

    train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    records = []
    for input, label in loader:
        if label[0].item() == label_val:
            records.append((input, label))

    del loader
    return records


def nonIID_CIFAR100(train_dir, t_id, seed, determinism):
    label_val = t_id + 1
    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])

    train_data = torchvision.datasets.CIFAR100(root=train_dir, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    records = []
    for input, label in loader:
        if label[0].item() == label_val:
            records.append((input, label))

    del loader
    return records


# for 1 label per-worker distribution over 10 workers
def bulkloader_noniid_CIFAR10(train_dir, t_id, world_size, train_bsz, input_shape, out_shape, seed, total_labels, determinism):
    valid_labels = []
    labels_perworker = total_labels // world_size
    for i in range(0,labels_perworker):
        valid_labels.append(labels_perworker * t_id + i)

    logging.info(f'VALID_LABELS for CIFAR-10 t_id {t_id} valid labels are {valid_labels}')
    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])

    train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)
    record = []
    for rec in train_data:
        if rec[1] in valid_labels:
            inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
            lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
            rec = (inp, lab)
            record.append(rec)

    del train_data
    trainloader = torch.utils.data.DataLoader(record, batch_size=train_bsz, shuffle=False)
    return trainloader, valid_labels


# for 4 labels per-worker distribution over 25 workers
def bulkloader_noniid_CIFAR100(train_dir, t_id, world_size, train_bsz, input_shape, out_shape, seed, total_labels, determinism):
    valid_labels = []
    labels_perworker = total_labels // world_size
    for i in range(0,labels_perworker):
        valid_labels.append(labels_perworker * t_id + i)

    logging.info(f'VALIDALABELS for t_id {t_id} valid labels are {valid_labels}')

    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])

    train_data = torchvision.datasets.CIFAR100(root=train_dir + 'data', train=True, download=True, transform=transform)
    record = []
    for rec in train_data:
        if rec[1] in valid_labels:
            inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
            lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
            record.append((inp, lab))

    del train_data
    trainloader = torch.utils.data.DataLoader(record, batch_size=train_bsz, shuffle=False)
    return trainloader, valid_labels


def perlabel_loaderCIFAR10(train_dir, bsz, input_shape, out_shape, seed, determinism, num_classes):
    perlabel_dataloader = {}
    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)
    for i in range(0,num_classes):
        record = []
        for rec in train_data:
            if rec[1] == i:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                rec = (inp, lab)
                record.append(rec)

        trainloader = torch.utils.data.DataLoader(record, batch_size=bsz, shuffle=True)
        perlabel_dataloader[i] = trainloader

    del train_data
    print(f'size of perlabel loader map {len(perlabel_dataloader)}')
    return perlabel_dataloader


def perlabel_loaderCIFAR100(train_dir, bsz, input_shape, out_shape, seed, determinism, num_classes):
    perlabel_dataloader = {}

    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR100(root=train_dir + 'data', train=True, download=True, transform=transform)

    for c in range(0, num_classes):
        record = []
        for rec in train_data:
            if rec[1] == c:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                record.append((inp, lab))

        trainloader = torch.utils.data.DataLoader(record, batch_size=bsz, shuffle=True)
        perlabel_dataloader[c] = trainloader

    return perlabel_dataloader
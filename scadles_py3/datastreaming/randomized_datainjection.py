import math
import random

import torch
import torchvision
import torchvision.transforms as transforms

import scadles_py3.misc.helper_fns as misc


def data_injection_CIFAR10(train_dir, t_id, train_bsz, input_shape, out_shape, seed, determinism, bsz, world_size,
                           num_labels, alpha, beta):
    valid_labels = [t_id]
    perlabel_dataloader = {}
    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)
    for i in range(0,10):
        record = []
        for rec in train_data:
            if rec[1] == i:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                rec = (inp, lab)
                record.append(rec)

        trainloader = torch.utils.data.DataLoader(record, batch_size=bsz, shuffle=True)
        perlabel_dataloader[i] = trainloader

    print('done creating a separate dataloader for each label cifar10!')
    record = []
    for rec in train_data:
        if rec[1] == t_id:
            inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
            lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
            rec = (inp, lab)
            record.append(rec)

    del train_data
    trainloader = torch.utils.data.DataLoader(record, batch_size=train_bsz, shuffle=True)
    for inp, lab in trainloader:
        injected_classes = int(math.ceil(alpha * world_size))
        loader_bsz = inp.size()[0]
        for _ in range(injected_classes):
            random_label = t_id
            while random_label in valid_labels:
                random_label = random.randint(0, num_labels-1)

            l1 = perlabel_dataloader[random_label]
            i1, la1 = next(iter(l1))
            batchsize1 = i1.size()[0]
            samplesize1 = int(beta * loader_bsz)
            if samplesize1 < batchsize1:
                i1, la1 = i1[:samplesize1], la1[:samplesize1]

            inp = torch.cat((inp, i1), dim=0)
            lab = torch.cat((lab, la1), dim=0)

        print(lab)
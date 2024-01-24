
import torch
import torchvision.models as models
from torch import optim, nn

import scadles_py3.misc.helper_fns as misc

# Models used for training are initialized here
def initialize_models(model_name, **kwargs):
    if model_name == 'vgg19':
        return VGG19model(lr=kwargs['lr'], momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'],
                              seed=kwargs['seed'], gamma=kwargs['gamma'], determinism=kwargs['determinism'])

    elif model_name == 'resnet152':
        return ResNet152model(lr=kwargs['lr'], momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'],
                              seed=kwargs['seed'], gamma=kwargs['gamma'], determinism=kwargs['determinism'])


class ResNet152model(object):

    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism=False):
        misc.set_seed(seed, determinism)
        self.model = models.resnet152(pretrained=False, progress=True)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        milestones = [75, 150, 225]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma,
                                                                 last_epoch=-1)

    def return_model(self):
        return self.model

    def return_optimizer(self):
        return self.optimizer

    def return_lrschedule(self):
        return self.lr_scheduler

    def return_lossfn(self):
        return self.loss_fn


class VGG19model(object):

    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism=False):
        misc.set_seed(seed, determinism)
        self.model = models.vgg19_bn(pretrained=False, progress=True)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        milestones = [75, 150, 200]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                 gamma=gamma, last_epoch=-1)

    def return_model(self):
        return self.model

    def return_optimizer(self):
        return self.optimizer

    def return_lrschedule(self):
        return self.lr_scheduler

    def return_lossfn(self):
        return self.loss_fn
#!/usr/bin/env python
# coding: utf-8

import math
import os
import itertools
from copy import deepcopy


os.environ["CUDA_VISIBLE_DEVICES"]="1"


from robustbench.utils import load_model
from robustbench.data import load_imagenet
from conf import cfg, load_cfg_fom_args
from robustbench.model_zoo.enums import ThreatModel
from robustbench.loaders import CustomImageFolder


import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from typing import Callable, Dict, Optional, Sequence, Set, Tuple


class SquaredAverageModel(nn.Module):
    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(SquaredAverageModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter +                     (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            squared_p_model_ = (p_model_**2) # squaring here
            if self.n_averaged == 0:
                p_swa.detach().copy_(squared_p_model_) 
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), squared_p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1


cfg.MODEL.ARCH = "Standard_R50"
cfg.CORRUPTION.DATASET = "imagenet"
cfg.CKPT_DIR = "./ckpt"

base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()


PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]),
    'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                   transforms.ToTensor()]),
    'none': transforms.Compose([transforms.ToTensor()]),
}


def load_imagenet_train(
        n_examples: Optional[int] = 5000,
        data_dir: str = './data',
        prepr: str = 'Res256Crop224') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_train = PREPROCESSINGS[prepr]
    # imagenet_train = CustomImageFolder(data_dir + '/ILSVRC2012_img_train', transforms_train, is_valid_file=False)
    
    imagenet_train = datasets.ImageFolder(root=data_dir + '/ILSVRC2012_img_train', transform=transforms_train)
    
    train_loader = data.DataLoader(imagenet_train, batch_size=n_examples,
                                  shuffle=True, num_workers=4)

    return imagenet_train, train_loader


batch_size = 128 # Using accumulation gradient, effective batch_size=256


imagenet_train, train_loader = load_imagenet_train(n_examples=batch_size, data_dir=cfg.DATA_DIR)


from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import trange
from tqdm import tqdm


n_batches = len(train_loader)
update_after = n_batches//4
extra_epochs = 2


optimizer = torch.optim.SGD(base_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()
swa_model = AveragedModel(base_model)
sqa_model = SquaredAverageModel(base_model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
swa_start = 90
swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=1, swa_lr=0.0001)

for epoch in range(swa_start, swa_start+extra_epochs):
    batch_idx = 0
    optimizer.zero_grad()
    for x_curr, y_curr in tqdm(train_loader):
        x_curr = x_curr.cuda()
        y_curr = y_curr.cuda()
        loss = loss_fn(base_model(x_curr), y_curr)
        loss.backward()
        if (batch_idx+1) % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx%update_after == 0:
                if epoch >= swa_start:
                    swa_model.update_parameters(base_model)
                    sqa_model.update_parameters(base_model)
                    swa_scheduler.step()
                else:
                    scheduler.step()


# Update bn statistics for the swa_model at the end (takes some time)
torch.optim.swa_utils.update_bn(train_loader, swa_model, device=torch.device("cuda"))


# Update bn statistics for the sqa_model at the end (takes some time)
torch.optim.swa_utils.update_bn(train_loader, sqa_model, device=torch.device("cuda"))


model_path = "./ckpt/imagenet/corruptions/Standard_R50_swa.pt"


torch.save(swa_model.module.state_dict(), model_path)


def covar(sqa_model, swa_model):
    cov_model = deepcopy(sqa_model)
    sqa_model = sqa_model
    swa_model = swa_model
    for p_cov, p_sqa, p_swa in zip(cov_model.parameters(), sqa_model.parameters(), swa_model.parameters()):
        p_sqa_ = p_sqa.detach()
        p_swa_ = p_swa.detach()
        p_cov.detach().copy_(p_sqa_ - (p_swa_**2))
    return cov_model


cov_model = covar(sqa_model, swa_model)


cov_model_path = "./ckpt/imagenet/corruptions/Standard_R50_cov.pt"
torch.save(cov_model.module.state_dict(), cov_model_path)

print("Files Standard_R50_cov.pt and Standard_R50_swa.pt created inside the directory ckpt/imagenet/corruptions/")

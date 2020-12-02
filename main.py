import numpy as np
import torch

from torchvision import transforms
import models
from models.model import*
from dataloaders import*
from configure import*
import torch.optim as optim

import argparse

import collections
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms.transforms import *

from method import wadg

import argparse

parser = argparse.ArgumentParser()



parser.add_argument('--target',type=str, help='The target domain A C P or S')
parser.add_argument("--lr_fea", type = float, help="learning_rate_fea", default=1e-5)
parser.add_argument("--lr_clf", type = float, help="learning_rate_clf", default=1e-5)
parser.add_argument("--lr_dis", type = float, help="learning_rate_dis", default=1e-5)
parser.add_argument("--weight_cls_loss", type = float, help="weight_cls_loss", default=1)
parser.add_argument("--weight_dis_loss", type = float, help="weight_dis_loss", default=1)





parser.add_argument("--weight_decay", type = float, help="weight_decay", default=1e-5)
parser.add_argument("--wd_round", type = int, help="learning_rate_dis", default=1)
parser.add_argument("--weight_mtr_loss", type = float, help="weight_mtr_loss", default=1e-5)

parser.add_argument("--mtr_margin", type = float, help="mtr_margin", default=1.0)
parser.add_argument("--mtr_scale_pos", type = float, help="mtr_loss_pos_scale", default=2.0)
parser.add_argument("--mtr_scale_neg", type = float, help="mtr_loss_neg_loss", default=40.0)
parser.add_argument("--gp_param", type = float, help="weight of gradient penalty", default=10.0)
parser.add_argument("--add_clsuter", type = int, help="weight of gradient penalty", default=5)


args = parser.parse_args()

print(args)

data_name = 'pacs'

config = all_configs[data_name]

params = {'fea_lr': args.lr_fea,
          'cls_lr':args.lr_clf,
        'dis_lr': args.lr_dis,
        'weight_decay': args.weight_decay,
        'batch_size':64,
        'w_d_round': args.wd_round,
        'weight_metric_loss':args.weight_mtr_loss,
        'weight_cls_loss':args.weight_cls_loss,
        'weight_dis_loss':args.weight_dis_loss,
        'gp_param':args.gp_param,
        'add_clsuter':args.add_clsuter
}

config.update(params)



config['param_metric'] = {'scale_pos':args.mtr_scale_pos,
                            'scale_neg':args.mtr_scale_neg,
                            'margin':args.mtr_margin}

target_ = args.target
config['target_name'] = target_
source_loaders, target_loader = return_dataset(data_name, target_, data_lists= data_lists, config = all_configs , mode= 'train', batch_size=config['batch_size'],need_balance = True)

FE = ResNet18Fc()

cls_net =CLS(512,config['num_classes'])
disnet = Alexnet_Discriminator_network(512)


strategy = wadg(net_fea = FE, net_clf = cls_net, net_dis=disnet,  source_loaders= source_loaders, target_loader=target_loader, args= config)
strategy.run()
print('============ Done ================')


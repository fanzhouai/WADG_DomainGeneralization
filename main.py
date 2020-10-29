import numpy as np
import torch
#from dataloaders import single_dataset, single_handler, source_target_dataset,source_target_handler,target_te_handler
#from model import Net_fea,Net_clf,Net_dis
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

from method import wass_dg

import argparse
#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()



#parser.add_argument('source',type=str, help='The source domain A D or W')
parser.add_argument('--target',type=str, help='The target domain A C P or S')
parser.add_argument("--lr_fea", type = float, help="learning_rate_fea", default=1e-5)
parser.add_argument("--lr_clf", type = float, help="learning_rate_clf", default=1e-5)
parser.add_argument("--lr_dis", type = float, help="learning_rate_dis", default=1e-5)
parser.add_argument("--lr_mtr", type = float, help="learning_rate_mtr", default=1e-5)
parser.add_argument("--weight_cls_loss", type = float, help="weight_cls_loss", default=1)
parser.add_argument("--weight_dis_loss", type = float, help="weight_dis_loss", default=1)





parser.add_argument("--weight_decay", type = float, help="weight_decay", default=1e-5)
parser.add_argument("--wd_round", type = int, help="learning_rate_dis", default=1)
parser.add_argument("--weight_mtr_loss", type = float, help="weight_mtr_loss", default=1e-4)

parser.add_argument("--mtr_margin", type = float, help="weight_mtr_loss", default=1.0)
parser.add_argument("--mtr_scale_pos", type = float, help="weight_mtr_loss", default=2.0)
parser.add_argument("--mtr_scale_neg", type = float, help="weight_mtr_loss", default=40.0)
parser.add_argument("--gp_param", type = float, help="weight of gradient penalty", default=10.0)
parser.add_argument("--add_clsuter", type = int, help="weight of gradient penalty", default=100)




#parser.add_argument("--gp_dis", type = float, default= 10,help='gradient-penalty coefficient in domain level critic')
#parser.add_argument('--gp_clf', type=float, default=1, help='gradient penalty in the classification loss') #
#parser.add_argument('--k_critic', type=int, default=5, help='Used to repeat some times for discriminator training')
#parser.add_argument('--k_clf', type=int, default=1, help='Used to repeat some times for clf training in adversarial training')
#parser.add_argument('--k-clf', type=int, default=1,help='Iterations for each bach round') 
#WDGRL use this --k-clf to enforce the source domain training, here we don't need to use it since we have already train about 20 epochs on source domain


args = parser.parse_args()

print(args)

data_name = 'pacs'

config = all_configs[data_name]

params = {'fea_lr': args.lr_fea,
          'cls_lr':args.lr_clf,
        'dis_lr': args.lr_dis,
        'metric_lr': args.lr_mtr,
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

save_threshold = {'A':81.0,
                'C':76.5,
                'S':77.0,
                'P':95.5}

config['param_metric'] = {'scale_pos':args.mtr_scale_pos,
                            'scale_neg':args.mtr_scale_neg,
                            'margin':args.mtr_margin}

target_ = args.target
config['target_name'] = target_
config['threshold'] = save_threshold[target_]
source_loaders, target_loader = return_dataset(data_name, target_, data_lists= data_lists, config = all_configs , mode= 'train', batch_size=config['batch_size'],need_balance = True)

#FE = AlexNetCaffeFc()
FE = ResNet18Fc()

cls_net =CLS(512,config['num_classes'])
disnet = Alexnet_Discriminator_network(512)
#mtr_net = Metric_net(512)



strategy = wass_dg(net_fea = FE, net_clf = cls_net, net_dis=disnet,  source_loaders= source_loaders, target_loader=target_loader, args= config)
strategy.train()
print('============ Done ================')


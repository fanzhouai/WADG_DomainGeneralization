from torchvision import transforms
import models
from models.model import*
from dataloaders import*
#from configs import*

import argparse

import collections
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms.transforms import *


datasets_dic = {'vlsc_datasets':["CALTECH", "LABELME", "PASCAL", "SUN"],
                'pacs_datasets':["art_painting", "cartoon", "photo", "sketch"]}

data_lists = {'vlsc':{'train':{'CALTECH':'./data/data_list/vlsc/CALTECH_train.txt',
                           'LABELME':'./data/data_list/vlsc/LABELME_train.txt',
                            'PASCAL':'./data/data_list/vlsc/PASCAL_train.txt',
                            'SUN':'./data/data_list/vlsc/SUN_train.txt'},
                            
                           'test':{'CALTECH':'./data/data_list/vlsc/CALTECH_test.txt',
                           'LABELME':'./data/data_list/vlsc/LABELME_test.txt',
                            'PASCAL':'./data/data_list/vlsc/PASCAL_test.txt',
                            'SUN':'./data/data_list/vlsc/SUN_test.txt'}},
                'pacs':{'train':{'art_painting':'./data/data_list/pacs/art_painting_train_kfold.txt',
                             'photo': './data/data_list/pacs/photo_train_kfold.txt',
                             'sketch':'./data/data_list/pacs/sketch_train_kfold.txt',
                             'cartoon':'./data/data_list/pacs/cartoon_train_kfold.txt'
                            },
                            'test':{'art_painting':'./data/data_list/pacs/art_painting_test_kfold.txt',
                             'photo': './data/data_list/pacs/photo_test_kfold.txt',
                             'sketch':'./data/data_list/pacs/sketch_test_kfold.txt',
                             'cartoon':'./data/data_list/pacs/cartoon_test_kfold.txt'
                            }
                            }}

all_configs = {'pacs':{'image_size':225,
          'train_transform':          
          transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
         'test_transform':
         transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
          'need_balance':True,
          'num_classes':7
         },
         'vlsc':{'image_size':225,
          'train_transform':          
          transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
         'test_transform':
         transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
          'need_balance':True,
          'num_classes':5
         }
         }
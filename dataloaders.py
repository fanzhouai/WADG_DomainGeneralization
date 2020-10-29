# Some of the loader functions were modified based on: https://github.com/thuml/Universal-Domain-Adaptation/blob/master/data.py
import pathlib
import os
from os.path import join
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms.transforms import *

from torchvision import utils
from PIL import Image
import random
import matplotlib.pyplot as plt
import collections
from collections import Counter

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler




def join_path(*a):
    return os.path.join(*a)
    
def return_dataset(data_name, target_, data_lists,config, mode= 'train', batch_size=16 ,need_balance = True, return_id = False, drop_last = True):
    #print(config)
    datasets = []
    if mode =='train':
        trans = config[data_name]['train_transform']
    elif mode in ['val','test']:
        trans = config[data_name]['test_transform']
    else:
        print('Invalid Mode. please use train, val or test')
    trans_t = config[data_name]['test_transform']
    
    
    if data_name in ['vlsc', 'VLSC']:
        all_domains = ["CALTECH", "LABELME", "PASCAL", "SUN"]
        _prefix = './data/vlsc/VLCS'
        num_class = 5
        if target_ == 'C':
            target_domain = 'CALTECH'
        elif target_ == 'L':
            target_domain = 'LABELME'
        elif target_ == 'V':
            target_domain = 'PASCAL'
        elif target_ == 'S':
            target_domain = 'SUN'
        else:
            print('Invalida domain and data combination')

    elif data_name in ['pacs', 'PACS']:
        print('use pacs dataset')
        all_domains = ["art_painting", "cartoon", "photo", "sketch"]
        _prefix = './data/PACS/kfold'
        num_class = 7
        if target_ == 'A':
            target_domain = 'art_painting'
        elif target_ == 'C':
            target_domain = 'cartoon'
        elif target_ == 'P':
            target_domain = 'photo'
        elif target_ == 'S':
            target_domain = 'sketch'
        else:
            print('Invalida domain and data combination')
    print('all domains ', all_domains)
    

    # First define the source and target domain
    print('target_domain is', target_domain)
    all_domains.remove(target_domain)
    source_domains = all_domains
    print('source_domains',source_domains)
    #print('source doamins', source_domains)

    for domain in source_domains:
        list_ = data_lists[data_name][mode][domain]
        datasets.append(FileListDataset(return_id = return_id, list_path= list_, path_prefix=_prefix,
                                        transform=trans, filter=(lambda x: x in range(num_class))))
    num_datasets = len(datasets)
    loaders = []
    
    for i in range(num_datasets):
        source_classes = datasets[i].labels
        source_freq = Counter(source_classes)
        source_class_weight = {x : 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
        source_weights = [source_class_weight[x] for x in datasets[i].labels]
        source_sampler = WeightedRandomSampler(source_weights, len(datasets[i].labels))
        loaders.append(DataLoader(datasets[i],batch_size=batch_size, sampler = source_sampler, drop_last= drop_last))
    
    target_list = data_lists[data_name]['test'][target_domain]
    target_dataset = FileListDataset(list_path= target_list, path_prefix=_prefix,transform=trans_t, filter=(lambda x: x in range(num_class)),return_id = return_id )

    target_loader = DataLoader(target_dataset,batch_size=batch_size, shuffle=True, drop_last=False)

    
    
    return loaders, target_loader
        



class To_load_Datasets:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


class BaseImageDataset(Dataset):
    """
    base image dataset
    for image dataset, ``__getitem__`` usually reads an image from a given file path
    the image is guaranteed to be in **RGB** mode
    subclasses should fill ``datas`` and ``labels`` as they need.
    """

    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x : x)
        self.datas = []
        self.labels = []

    def __getitem__(self, index):
        
        
        idx = index
        im = Image.open(self.datas[index]).convert('RGB')
        im = self.transform(im)
    

        if not self.return_id:
            return im, self.labels[idx]
        return im, self.labels[idx], idx

    def __len__(self):
        return len(self.datas)

        
            
class FileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1])-1 for x in data] # since the label starts from 1 so we need minus 1
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1
        
        def return_datas(self):
            
            return self.datas

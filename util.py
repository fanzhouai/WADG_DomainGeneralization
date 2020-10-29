import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

def compute_wasserstein(features, btch_sz, feature_extractor, discriminator, use_gp = True, gp_weight = 0.1):

    num_domains = int(len(features)/btch_sz)
    dis_loss = 0
    for t in range(num_domains):

        for k in range(t + 1, num_domains):
            
            features_t = features[t * btch_sz:(t + 1) * btch_sz]
            features_k = features[k * btch_sz:(k + 1) * btch_sz]
            
            dis_t = discriminator(features_t)
            dis_k = discriminator(features_k)
            
            if use_gp:
                gp = gradient_penalty(discriminator, features_t, features_k)
                disc_loss = dis_t.mean() - dis_k.mean() - gp_weight*gp
            else: 
                disc_loss = dis_t.mean() - dis_k.mean()

            #gradient_pntly=self.gradient_penalty(inputs[t * btch_sz:(t + 1) * btch_sz],inputs[k * btch_sz:(k + 1) * btch_sz], t, k)
            # critic loss --->  E(f(x)) - E(f(y)) + gamma* ||grad(f(x+y/2))-1||

            #(pred_t.mean() - pred_k.mean() ) + self.grad_weight *gradient_pntly
                        #  negative sign compute wasserstien distance
            #discrm_distnc_mtrx[t, k] += -(pred_t.mean() - pred_k.mean()).item()
            #discrm_distnc_mtrx[k, t] = discrm_distnc_mtrx[t, k]

            dis_loss += disc_loss
    
    return dis_loss

# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


# setting gradient penalty for sure the lipschitiz property
def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty

def _return_dataset(data_name, target_, data_lists,config, mode= 'train', batch_size=16 ,need_balance = True):
    #print(config)
    datasets = []
    if mode in ['train','val']:
        trans = config[data_name]['train_transform']
    elif mode == 'test':
        trans = config[data_name]['test_transform']
    else:
        print('Invalid Mode. please use train, val or test')
    
    if data_name in ['vlsc', 'VLSC']:
        all_domains = ["CALTECH", "LABELME", "PASCAL", "SUN"]
        _prefix = './data/vlsc/VLSC'
        num_class = 5
        if target_ == 'C':
            target_domain = 'CALTECH'
        elif target_ == 'L':
            target_domain = 'LABELME'
        elif target_ == 'P':
            target_domain == 'PASCAL'
        elif target_ == 'S':
            target_domain == 'SUN'
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

    all_domains.remove(target_domain)
    source_domains = all_domains
    print('source_domains',source_domains)
    #print('source doamins', source_domains)

    for domain in source_domains:
        list_ = data_lists[data_name][mode][domain]
        datasets.append(FileListDataset(list_path= list_, path_prefix=_prefix,
                                        transform=trans, filter=(lambda x: x in range(num_class))))
    num_datasets = len(datasets)
    loaders = []
    
    for i in range(num_datasets):
        source_classes = datasets[i].labels
        source_freq = Counter(source_classes)
        source_class_weight = {x : 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
        source_weights = [source_class_weight[x] for x in datasets[i].labels]
        source_sampler = WeightedRandomSampler(source_weights, len(datasets[i].labels))
        loaders.append(DataLoader(datasets[i],batch_size=batch_size, sampler = source_sampler, drop_last=True))
    
    target_list = data_lists[data_name]['test'][target_domain]
    target_dataset = FileListDataset(list_path= target_list, path_prefix=_prefix,transform=trans, filter=(lambda x: x in range(num_class)))
    
    #target_classes = target_dataset.labels
    #target_freq = Counter(target_classes)
    #target_class_weight = {x : 1.0 / target_freq[x] if need_balance else 1.0 for x in target_freq}

    #target_weights = [target_class_weight[x] for x in target_dataset.labels]
    #target_sampler = WeightedRandomSampler(target_weights, len(target_dataset.labels))
    target_loader = DataLoader(target_dataset,batch_size=batch_size, shuffle=True, drop_last=True)

    
    
    return loaders, target_loader


class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg['scale_pos']
        self.scale_neg = cfg['scale_neg']

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        #print('batch_size is', batch_size)
        sim_mat = torch.matmul(feats, torch.t(feats))
        #print('lables is',labels)
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

def sort_loaders(loaders_list, reverse=True):
    num_loaders = len(loaders_list)
    
    tuple_ = []
    
    for i in range(num_loaders):
        tuple_.append((loaders_list[i],len(loaders_list[i])))
    
    return sorted(tuple_, key=lambda tuple_len: tuple_len[1],reverse=reverse)
    

def get_optimizer(model, init_lr, momentum, weight_decay, feature_fixed=False, nesterov=False, per_layer=False):
    if feature_fixed:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        if per_layer:
            if not isinstance(model, list):
                raise ValueError('Model must be a list type.')
            optimizer = optim.SGD(
                [{'params': model_.parameters(), 'lr': init_lr*alpha} for model_, alpha in model],
                lr=init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
                                   
        else:
            params_to_update = model.parameters()
            optimizer = optim.SGD(
                params_to_update, lr=init_lr, momentum=momentum, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    return optimizer
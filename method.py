import torch

from dataloaders import*
from util import*
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
from tqdm import tqdm
from util import MultiSimilarityLoss, sort_loaders
import lr_schedule
import statistics
from time import gmtime, strftime

class wadg():
    def __init__(self, net_fea, net_clf, net_dis, source_loaders, target_loader, args):
        
        self.target_name = args['target_name']

        self.fea_lr = args['fea_lr']
        self.cls_lr = args['cls_lr']
        self.dis_lr = args['dis_lr']


        self.source_loaders = source_loaders
        self.target_loader = target_loader

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.fea = net_fea.to(self.device)
        self.clf = net_clf.to(self.device)
        self.dis = net_dis.to(self.device)

        self.args = args
        self.weight_cls_loss = self.args['weight_cls_loss']
        self.weight_dis_loss = self.args['weight_dis_loss']

        self.weight_decay = self.args['weight_decay']
        self.batch_size = self.args['batch_size']
        self.w_d_round = self.args['w_d_round']
        param_metric = self.args['param_metric']
        self.multi_similarity_loss = MultiSimilarityLoss(param_metric)
        self.weight_metric_loss = self.args['weight_metric_loss']
        self.gp_param = self.args['gp_param']
        self.add_clsuter = self.args['add_clsuter']

    def run(self):

        self.fea.train()
        self.clf.train()             
        self.dis.train() 

        all_acc = []


        best_acc = 0
        total_epoch = 40
        time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print('training start at ',time_now)
        for epoch in range(total_epoch):
            
            runing_dis_loss = []
            runing_cls_loss = []
            runing_mtr_loss = []
            if (epoch+1)%15 ==0:
                self.fea_lr = self.fea_lr*0.5
                self.dis_lr = self.dis_lr*0.5
                self.cls_lr = self.cls_lr*0.5
                self.weight_cls_loss =self.weight_cls_loss*2.0
                self.weight_dis_loss =self.weight_dis_loss*1.1


            opt_fea = optim.Adam(self.fea.parameters() , lr=self.fea_lr)
            opt_clf = optim.Adam(self.clf.parameters(), lr=self.cls_lr,weight_decay=self.weight_decay)
            opt_dis = optim.Adam(self.dis.parameters(), lr=self.dis_lr)
            opt_fea.zero_grad()
            opt_clf.zero_grad()
            opt_dis.zero_grad()
            
            sort_loaders_list = sort_loaders(self.source_loaders)
            batches = iter(sort_loaders_list[0][0])


            #batches = zip(self.source_loaders[0], self.source_loaders[1], self.source_loaders[2])
            num_batches = len(sort_loaders_list[0][0])
            iter_t = iter(self.target_loader)
            i = 0
            total_acc_t = 0
            
            for (x1,y1) in tqdm(batches, leave=False, total=num_batches):
                self.fea.to(self.device)
                self.clf.to(self.device)
                self.dis.to(self.device)

                        
                p = float(i + (epoch+1)* num_batches) / (total_epoch )/ num_batches
                trade_off = 2. / (1. + np.exp(-10 * p)) - 1
            
                if i % len(sort_loaders_list[1][0]) == 0:
                    iter2 = iter(sort_loaders_list[1][0])
                
                if i % len(sort_loaders_list[2][0]) == 0:
                    iter3 = iter(sort_loaders_list[2][0])
                i+=1


                x2, y2 = next(iter2)
                x3, y3 = next(iter3)

                x1, y1, x2, y2, x3, y3 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda(),  x3.cuda(), y3.cuda()
                
                source_images = torch.cat((x1, x2, x3),0)
                source_labels = torch.cat((y1, y2, y3),0)
                source_images.to(self.device)
                source_labels.to(self.device)


                # we shall train the feature extractor and classifier
                self.fea.train()
                self.clf.train()
                self.dis.eval()

                set_requires_grad(self.fea, requires_grad=True)
                set_requires_grad(self.clf, requires_grad=True)
                set_requires_grad(self.dis, requires_grad=False)
                
                source_fea = self.fea(source_images)
                
                _, fc1_s, fc2_s, predict_prob_source = self.clf(source_fea)

                ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, source_labels)
                cls_loss = torch.mean(ce, dim=0, keepdim=True)

                # Then, we can compute the wasserstein distance
                images_source = source_images # Recompute one to avoild gradient problems

                fea_source = self.fea(images_source)
                dis_loss = compute_wasserstein(fea_source,  btch_sz = self.batch_size,
                                                feature_extractor = self.fea, discriminator=self.dis ,use_gp= True, gp_weight= self.gp_param)
                
                if epoch>self.add_clsuter:
                                    #mtr_out = self.mtr(fea_source)
                    mtr_out = F.normalize(fc1_s, p=2, dim=1)
                    mtr_loss = self.multi_similarity_loss(mtr_out,source_labels)
                    runing_mtr_loss.append(mtr_loss.item())
                    loss = cls_loss +trade_off*dis_loss + self.weight_metric_loss* mtr_loss 
                else:
                    loss = cls_loss +trade_off*dis_loss # + self.weight_metric_loss* mtr_loss 
                    runing_mtr_loss.append(0)

                runing_cls_loss.append(cls_loss.item())
                loss.backward()
                opt_fea.step()
                opt_clf.step()  

                     

                self.fea.eval()
                self.clf.eval()
                self.dis.train()

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)                      
                set_requires_grad(self.dis, requires_grad=True)
                        
                with torch.no_grad():

                    z_s = self.fea(source_images)


                for _ in range(self.w_d_round):

                    dis_s_t_loss = -1.0*self.weight_dis_loss*compute_wasserstein(z_s, btch_sz= self.batch_size, 
                                                feature_extractor=self.fea, discriminator= self.dis, use_gp=True, gp_weight=self.gp_param)

                    
                    runing_dis_loss.append(dis_s_t_loss.item())

                    dis_s_t_loss.backward()
                    opt_dis.step() # 

                opt_fea.zero_grad()
                opt_clf.zero_grad()
                opt_dis.zero_grad()

           
            print('=========testing=============')
            num_batches = len(self.target_loader)
            total_acc_t = 0

            iter_t = iter(self.target_loader)

            for x_t, y_t in tqdm(iter_t,leave=False, total=len(self.target_loader)):
                x_t, y_t = x_t.cuda(), y_t.cuda()
                            
                # Then we shall test the test results on the target domain
                self.fea.eval()
                self.clf.eval()
                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)                      
                set_requires_grad(self.dis, requires_grad=False)

                    
                with torch.no_grad():
                        
                    latent = self.fea(x_t)
                    _,_,_, out1 = self.clf(latent)

                total_acc_t    += (out1.max(1)[1] == y_t).float().mean().item()

            all_acc.append(acc_t)

            print('========== epoch {:d} ========'.format(epoch))
            print('     Runing cls loss is ', statistics.mean(runing_cls_loss))
            print('     Runing dis loss is ', statistics.mean(runing_dis_loss))
            print('     Runing mtr loss is ', statistics.mean(runing_mtr_loss))
            print('     Mean acc on target domain is ', acc_t)



        print('best acc is', max(all_acc))
        print('train model index',time_now)
        return all_acc
             

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

class wass_dg():
    def __init__(self, net_fea, net_clf, net_dis, source_loaders, target_loader, args):
        
        self.target_name = args['target_name']

        self.fea_lr = args['fea_lr']
        self.cls_lr = args['cls_lr']
        self.dis_lr = args['dis_lr']
        self.metric_lr = args['metric_lr']

        #self.forward_metric_net = Metric_net


        self.source_loaders = source_loaders
        self.target_loader = target_loader

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.fea = net_fea.to(self.device)
        self.clf = net_clf.to(self.device)
        self.dis = net_dis.to(self.device)
        #self.mtr = mtr_net.to(self.device)

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
        self.save_threshold = self.args['threshold']

        
        #####################333
        
        ########################
     

        


    def train(self):

        self.fea.train()
        self.clf.train()             
        self.dis.train() # dis_s_t is used for domain level critic

        
        #opt_mtr = optim.Adam(list(self.mtr.parameters()) + list(self.mtr.parameters()), lr= self.metric_lr)


        #optimizer_fea = optim.SGD(self.fea.parameters(), lr=self.fea_lr, momentum=0.9)
        #optimizer_clf = optim.SGD(self.clf.parameters(), lr=self.cls_lr, momentum=0.9,weight_decay=self.weight_decay)
        #optimizer_dis = optim.SGD(self.dis.parameters(), lr=self.dis_lr, momentum=0.9)
        '''
        fea_parameter_list = self.fea.get_parameters()
        optimizer_fea_config = self.args["optimizer_fea"]
        optimizer_fea = optimizer_fea_config['type'](fea_parameter_list, \
                        **(optimizer_fea_config["optim_params"]))
        param_lr_fea = []
        for param_group in optimizer_fea.param_groups:
            param_lr_fea.append(param_group["lr"])
        schedule_param_fea = optimizer_fea_config["lr_param"]
        fea_lr_scheduler = lr_schedule.schedule_dict[optimizer_fea_config["lr_type"]]
        
        clf_parameter_list = self.clf.get_parameters()
        optimizer_clf_config = self.args["optimizer_clf"]
        optimizer_clf = optimizer_clf_config["type"](clf_parameter_list, \
                        **(optimizer_clf_config["optim_params"]))
        param_lr_clf = []
        for param_group in optimizer_clf.param_groups:
            param_lr_clf.append(param_group["lr"])
        schedule_param_clf = optimizer_clf_config["lr_param"]
        clf_lr_scheduler = lr_schedule.schedule_dict[optimizer_clf_config["lr_type"]]
        ######
        dis_parameter_list = self.dis.get_parameters()
        optimizer_dis_config = self.args["optimizer_dis"]
        optimizer_dis = optimizer_dis_config["type"](dis_parameter_list, \
                        **(optimizer_dis_config["optim_params"]))
        param_lr_dis = []
        for param_group in optimizer_dis.param_groups:
            param_lr_dis.append(param_group["lr"])
        schedule_param_dis = optimizer_dis_config["lr_param"]
        dis_lr_scheduler = lr_schedule.schedule_dict[optimizer_dis_config["lr_type"]]
        ##########3
        
        mtr_parameter_list = self.mtr.get_parameters()
        optimizer_mtr_config = self.args["optimizer_mtr"]
        optimizer_mtr = optimizer_mtr_config["type"](mtr_parameter_list, \
                        **(optimizer_mtr_config["optim_params"]))
        param_lr_mtr = []
        for param_group in optimizer_mtr.param_groups:
            param_lr_mtr.append(param_group["lr"])
        schedule_param_mtr = optimizer_mtr_config["lr_param"]
        mtr_lr_scheduler = lr_schedule.schedule_dict[optimizer_mtr_config["lr_type"]]
        '''



        #total_step = 2000
        #epoch_list = [20,40,60, 80]

        #opt_fea = optim.lr_scheduler.MultiStepLR(optimizer_fea, milestones=epoch_list, gamma=0.5)
        #opt_clf = optim.lr_scheduler.MultiStepLR(optimizer_clf, milestones=epoch_list, gamma=0.5)
        #opt_dis = optim.lr_scheduler.MultiStepLR(optimizer_dis, milestones=epoch_list, gamma=0.5)
        '''

        '''

        #opt_fea = optim.Adam(self.fea.parameters() , lr=1e-5)
        #opt_clf = optim.Adam(self.clf.parameters(), lr=1e-5,weight_decay =self.weight_decay)
        #opt_dis_s_t = optim.Adam(self.dis_s_t.parameters(), lr=1e-5)
        all_acc = []
        
        #trade_off = 1.0

        #for step in range(total_step):

        '''
            opt_fea = fea_lr_scheduler(optimizer_fea, step, **schedule_param_fea)
            opt_clf = clf_lr_scheduler(optimizer_clf, step, **schedule_param_clf)
            opt_dis = dis_lr_scheduler(optimizer_dis, step, **schedule_param_dis)
            #opt_mtr = dis_lr_scheduler(optimizer_mtr, step, **schedule_param_mtr)
        '''

            


            #p = step / total_step
            #trade_off = (2. / (1. + np.exp(-10 * p)) -1) * self.dis_loss_weight
            #beta = (2. / (1. + np.exp(-10 * p)) -1) * entropy_weight
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
                #loss.to(self.device)
                loss.backward()
                #optimizer_fea.step()
                #optimizer_clf.step()
                opt_fea.step()
                opt_clf.step()  
                #opt_mtr.step()

                # Then, we can compute the wasserstein distance
                # Here we don't use fc1_s but to compute source_z again to avoid potentail variable gradient conflict issue
                #source_z = self.fea(x_s)
                #target_z = self.fea(x_t)

                # Then compute the wasserstein distance between source and target
                #wassertein_source_and_target = self.dis_s_t(source_z).mean() - self.dis_s_t(target_z).mean()
                 # Then compute the gradient penalty
                #gp_s_t = gradient_penalty(self.dis_s_t, target_z, source_z)
                            
                # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
                #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
                #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)

                      

                self.fea.eval()
                self.clf.eval()
                self.dis.train()

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)                      
                set_requires_grad(self.dis, requires_grad=True)
                        
                with torch.no_grad():

                    z_s = self.fea(source_images)


                for _ in range(self.w_d_round):

                    # gradient ascent for multiple times like GANS training

                    #gp_s_t = gradient_penalty(self.dis_s_t, z_s, z_t)

                    #wassertein_source_and_target = self.dis_s_t(z_s).mean() - self.dis_s_t(z_t).mean()
                    dis_s_t_loss = -1*self.weight_dis_loss*compute_wasserstein(z_s, btch_sz= self.batch_size, 
                                                feature_extractor=self.fea, discriminator= self.dis, use_gp=True, gp_weight=self.gp_param)
                    #dis_s_t_loss = (-1 * wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                    # Currently we don't apply any weights on w-disstance loss
                    
                    runing_dis_loss.append(dis_s_t_loss.item())

                    dis_s_t_loss.backward()
                    #optimizer_dis.step()
                    opt_dis.step() # 

                opt_fea.zero_grad()
                opt_clf.zero_grad()
                opt_dis.zero_grad()

           
            print('=========testing=============')
            num_batches = len(self.target_loader)
            total_acc_t = 0
            #trade_off+=0.03
            #self.fea_lr = self.fea_lr*0.999
            #self.cls_lr = self.cls_lr*0.999
            #self.dis_lr = self.dis_lr*0.999



            iter_t = iter(self.target_loader)

            for x_t, y_t in tqdm(iter_t,leave=False, total=len(self.target_loader)):
                x_t, y_t = x_t.cuda(), y_t.cuda()
                            
                    # Then we shall test the test results on the target domain
                self.fea.eval()
                self.clf.eval()
                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)                      
                set_requires_grad(self.dis, requires_grad=False)
                    #set_requires_grad(self.mtr, requires_grad=False)


                    
                with torch.no_grad():
                        
                    latent = self.fea(x_t)
                    _,_,_, out1 = self.clf(latent)

                total_acc_t    += (out1.max(1)[1] == y_t).float().mean().item()
                
            acc_t = 100.0* total_acc_t/num_batches
            if acc_t > best_acc:
                best_acc = acc_t
                if best_acc >  self.save_threshold:
                    torch.save(self.fea,'./saved_models_res18/fea'+self.target_name+time_now)
                    torch.save(self.clf,'./saved_models_res18/clf'+self.target_name+time_now)

            all_acc.append(acc_t)

            print('========== epoch {:d} ========'.format(epoch))
            print('     Runing cls loss is ', statistics.mean(runing_cls_loss))
            print('     Runing dis loss is ', statistics.mean(runing_dis_loss))
            print('     Runing mtr loss is ', statistics.mean(runing_mtr_loss))
            print('     Mean acc on target domain is ', acc_t)



        print('best acc is', max(all_acc))
        print('train model index',time_now)
        return all_acc
             

        

    

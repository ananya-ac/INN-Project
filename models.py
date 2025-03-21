

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import torch.nn as nn
import config 
from loader import get_loaders, cubic_equation_loaders, get_GI_loaders
import torch
from loss_functions import MMD_multiscale, fit_l2, normal_prior_x_loss , uniform_prior_loss, NLL
import pytorch_lightning as pl
from loader import get_test
from typing import Tuple
import torch.utils.data as data
from geomloss import SamplesLoss
import pdb

approx_wasserstein = SamplesLoss(blur=0.01)


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))


def distance_euclidean(pos_target: torch.FloatTensor, pos: torch.FloatTensor) -> float:
        pdist = torch.nn.PairwiseDistance(p=2)
        dim = pos.shape[0]
        pos = pos.to(pos_target.device)
        return torch.sum(pdist(pos_target, pos)) / dim

def advance_joint(current_pos: torch.FloatTensor, length: float, angle: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        next_pos = current_pos.clone()
        next_pos[:, 0] = next_pos[:, 0] + length * torch.cos(angle)
        next_pos[:, 1] = next_pos[:, 1] + length * torch.sin(angle)
        return current_pos, next_pos



def get_inn(dim_tot):
    nodes = [InputNode(dim_tot , name='input')]

    for k in range(config.num_layers):
        nodes.append(Node(nodes[-1],
                        GLOWCouplingBlock,
                        {'subnet_constructor':subnet_fc, 'clamp':2.0},
                        name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                        PermuteRandom,
                        {'seed':k},
                        name=F'permute_{k}'))

    nodes.append(OutputNode(nodes[-1], name='output'))
    
    model =  ReversibleGraphNet(nodes, verbose=False)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    
    for param in trainable_parameters:
        param.data = 0.05*torch.randn_like(param)
    
    return model



class INNModel(pl.LightningModule):
    def __init__(self, prior, loss_f, dataset = 'ik_robotics'):
        super(INNModel, self).__init__()
        self.loss_factor = 1.0
        self.i_epoch = 0
        self.prior = prior
        self.loss_f = loss_f
        self.dataset = dataset
        if dataset == 'cubic':
            self.c = config.Cubic_Config
        if dataset == 'ik_robotics':
            self.c = config.IK_Config
        if dataset == 'GI':
            self.c = config.GI_Config
        self.model = get_inn(self.c.ndim_tot)
        
        

    def forward(self, x, rev = False):
        if rev: return self.model(x, rev = rev)
        else: return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        if self.c.ndim_y == 1:
            y = y.unsqueeze(1)
        if self.c.ndim_x == 1:
            x = x.unsqueeze(1)

        y_clean = y.clone()
        
        if self.dataset.lower() != 'gi':
            y = y + config.y_noise_scale * torch.randn_like(y, dtype=torch.float, device=config.device)
        
        x = torch.cat([x, config.zeros_noise_scale * torch.randn(config.batch_size, self.c.ndim_tot - self.c.ndim_x, device=config.device)], dim=1)
        y = torch.cat([y, config.y_noise_scale * torch.randn(config.batch_size, self.c.ndim_z, device=config.device)], dim=1)
        
        output_rev, _ = self.model(y, rev=True)
        output, sum_log_j_f = self.model(x)
        

        y_act = y_clean[:, :self.c.ndim_y]
        y_preds = output[:, :self.c.ndim_y]
        z_preds = output[:, self.c.ndim_y:]
        
        y_clean = torch.cat([y_clean, y[:, self.c.ndim_y:]], dim=1)
        
        if self.loss_f == 'NLL':
            loss = (1/config.var) * fit_l2(y_preds, y_act)
            loss += NLL(z_preds, sum_log_j_f)
        
        if self.loss_f == 'MMDf' or self.loss_f == 'MMDa':
            loss = 100 * MMD_multiscale(y_clean,output, kind='forward')
            loss +=  3 * fit_l2(y_preds, y_act)
            
        if self.loss_f == 'wasserstein':
            loss = 3 * fit_l2(y_preds, y_act)

        
        
        l_tot = loss
        
        loss_backward = 0
        
        
        if self.loss_f == 'wasserstein':
            #loss_backward += approx_wasserstein(x[:, :self.ndim_x], output_rev[:, :self.ndim_x])
            loss_backward += approx_wasserstein(x, output_rev)
        
        
        if self.loss_f == 'MMDa':
            loss_backward = self.loss_factor * 500 * MMD_multiscale(x, output_rev)
            #loss_backward = self.loss_factor * 500 * MMD_multiscale(x[:, :self.ndim_x], output_rev[:, :self.ndim_x])
            loss_backward += fit_l2(output_rev, x)
        
        if self.prior:
            if self.dataset == 'ik_robotics':
                loss_backward += config.prior_scale * normal_prior_x_loss(x_gt=x[:, :self.c.ndim_x], x = output_rev[:, :self.c.ndim_x]) 
                #loss_backward += config.prior_scale * uniform_prior_loss(output_rev[:, :self.c.ndim_x])
            else : loss_backward += uniform_prior_loss(output_rev[:, :self.c.ndim_x])
        
        l_tot += loss_backward
        self.loss_factor = min(1., 2. * 0.002**(1. - (float(self.i_epoch) / 3120)))
        self.i_epoch += 1 


        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-15.00, 15.00)
                
        

        self.log("train_loss", l_tot)    
            

        return l_tot
    
    #comment out for shell script runs
    
    # def validation_step(self, batch, batch_idx):
        
    #     if self.dataset == 'ik_robotics':
    #         y_test = get_test()
    #         generated_x = [self(torch.cat([y_test.unsqueeze(0).to(config.device), config.y_noise_scale * torch.randn(size = (1, self.c.ndim_tot - self.c.ndim_y)).to(config.device)], dim = 1), rev = True)[0] for i in range(config.num_samples)]
    #         thetas = torch.stack(generated_x).squeeze(1)[:, :self.c.ndim_x].to(config.device)
    #         pos = y_test.repeat(config.num_samples,1)
    #         angle = torch.zeros_like(thetas[:, 1], device=thetas.device)
    #         p_next = torch.stack([torch.zeros((thetas.shape[0]), device=thetas.device), thetas[:, 0]], axis=1)
    #         lengths = torch.tensor(self.c.lengths, device=config.device)
            
    #         for joint in range(self.c.num_joints -1):
    #             # Advance one joint
    #             angle += thetas[:, joint + 1]
    #             _, p_next = advance_joint(p_next, lengths[joint], angle)
                
    #             # Calculate distance from target
    #         distance = distance_euclidean(pos, p_next)
    #         self.log("mean_euclidean_distance", distance)
    #     elif self.dataset == 'cubic':
    #         x,y = batch
    #         if self.c.ndim_y == 1:
    #             y = y.unsqueeze(1)
    #         if self.c.ndim_x == 1:
    #             x = x.unsqueeze(1)

    #         y += config.y_noise_scale * torch.randn_like(y)
    #         y = torch.cat([y, config.y_noise_scale * torch.randn(y.shape[0], self.c.ndim_z, device=config.device)], dim=1)
    #         output_rev, _ = self.model(y, rev=True)
    #         loss = fit_l2(x[:, :self.c.ndim_x], output_rev[:, :self.c.ndim_x])
    #         self.log("val_loss", loss)
        
    #     else:
    #         x,y = batch
    #         y = torch.cat([y, torch.randn(y.shape[0], self.c.ndim_z, device=config.device)], dim=1)
    #         output_rev, _ = self.model(y, rev=True)
    #         loss = fit_l2(x[:, :self.c.ndim_x], output_rev[:, :self.c.ndim_x])
    #         self.log("val_loss", loss)
        
            
    
    def val_dataloader(self):
        if self.dataset.lower() == 'ik_robotics':
            _, val_loader = get_loaders()
            
        elif self.dataset.lower() == 'cubic':
            _, val_loader = cubic_equation_loaders(100000)
        
        else:
            _, val_loader = get_GI_loaders()    
    

        return val_loader

    def configure_optimizers(self):
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.l2_reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',  # Monitor the training loss (optional, depends on the scheduler type)
                'interval': 'epoch',  # Scheduler step happens every epoch
                'frequency': 1,  # How often the scheduler step happens
            }
        }

    def train_dataloader(self):
        if self.dataset == 'cubic':
            train_loader, _ = cubic_equation_loaders(100000)
            
        elif self.dataset == 'ik_robotics':
            train_loader, _ = get_loaders()
        
        else:
            train_loader, _ = get_GI_loaders()
        
        return train_loader

    
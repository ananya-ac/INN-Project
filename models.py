import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple
from geomloss import SamplesLoss

import config
from loader import get_loaders, get_GI_loaders, get_test
from loss_functions import fit_l2, normal_prior_x_loss, uniform_prior_loss, NLL
from networks import get_inn, get_iresnet


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


class INNModel(pl.LightningModule):
    def __init__(self, prior, loss_f, dataset, hidden_units, entropy_reg, num_layers, latent_dim, lr, network_type, prior_level, prior_type):
        super(INNModel, self).__init__()

        self.i_epoch = 0
        self.prior = prior
        self.loss_f = loss_f
        self.dataset = dataset
        self.prior_level = prior_level
        self.prior_type = prior_type
        self.loss_weights = config.LossWeightsConfig()
        if dataset.lower() == 'ik_robotics':
            self.c = config.IKConfig()
        elif dataset.lower() == 'gi':
            self.c = config.GIConfig()
            
        
        if loss_f == 'wasserstein':
            self.approx_wasserstein = SamplesLoss(p = 2, blur = entropy_reg)

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
        
        self.ndim_tot = self.c.ndim_y + latent_dim
        self.latent_dim = latent_dim
        print("Total dim:", self.ndim_tot)
        if network_type == 'glow':
            self.model = get_inn(self.ndim_tot, hidden_units=hidden_units, num_layers=num_layers).to(config.device)
        else:
            self.model = get_iresnet(self.ndim_tot, num_blocks=num_layers, hidden_size=hidden_units).to(config.device)
        self.lr = lr

    def forward(self, x, rev = False):
        
        
        
        if rev:
            out, jac = self.model(x, rev = rev)
            return out, jac 
        else: 
            out, jac_inv = self.model(x)
            return out, jac_inv  
        
    def sample(self, y, n_samples=50):
        """
        Generate samples from the model given conditioning input y.

        Args:
            y (Tensor): shape (batch_size, y_dim)
            n_samples (int): number of samples per input

        Returns:
            sample (Tensor): shape (batch_size, n_samples, x_dim)
        """
        batch_size, y_dim = y.shape
        
        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std

        # Repeat y n_samples times
        y_rep = y.unsqueeze(1).expand(batch_size, n_samples, y_dim)  # (B, n, D)
        y_rep = y_rep.contiguous().view(-1, y_dim)  # (B * n, D)

        # Sample noise
        noise = torch.randn(batch_size * n_samples, self.latent_dim, device=config.device)

        # Concatenate y and noise
        y_full = torch.cat([y_rep, noise], dim=1)

        # Pass through inverse model
        sample, _ = self.model(y_full, rev=True)

        # Undo normalization
        if self.x_mean is not None:
            sample = sample[:, :self.c.ndim_x] * self.x_std + self.x_mean

        # Reshape back to (batch_size, n_samples, x_dim)
        sample = sample.view(batch_size, n_samples, -1)

        return sample

    
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        batch_size = x.shape[0]
        if self.c.ndim_y == 1:
            y = y.unsqueeze(1)
        if self.c.ndim_x == 1:
            x = x.unsqueeze(1)

        y_clean = y.clone()

        if self.dataset.lower() != 'gi':
            y = y + config.y_noise_scale * torch.randn_like(y, dtype=torch.float, device=config.device)

        # Construct x tensor with padding
        pad_x = config.zeros_noise_scale * torch.randn(batch_size, self.ndim_tot - self.c.ndim_x, device=config.device)
        x = torch.cat([x, pad_x], dim=1)
        # Construct yz tensor as [z, pad, y]
        z = torch.randn(batch_size, self.latent_dim, device=config.device)
        pad_yz = config.zeros_noise_scale * torch.randn(batch_size, self.ndim_tot - self.c.ndim_y - self.latent_dim, device=config.device)
        y = torch.cat([z, pad_yz, y], dim=1)

        
        output_rev, _ = self.model(y, rev=True)
        output, sum_log_j_f = self.model(x)

        y_act = y_clean[:, :self.c.ndim_y]
        y_preds = output[:, -self.c.ndim_y:]
        z_preds = output[:, :self.latent_dim]
        y_clean_full = torch.cat([z, y_clean], dim=1)
        
        if self.loss_f == 'NLL':
            loss = (self.loss_weights.nll_fit_l2_weight * fit_l2(y_preds, y_act) +
                    NLL(z_preds, sum_log_j_f) +
                    self.loss_weights.nll_reconstruction_weight * fit_l2(output_rev, x))

        elif self.loss_f == 'wasserstein':
            loss = self.loss_weights.wasserstein_fit_l2_weight * fit_l2(y_preds, y_act)

        l_tot = loss
        loss_backward = 0

        if self.loss_f == 'wasserstein':
            loss_backward += self.loss_weights.wasserstein_reconstruction_weight * fit_l2(output_rev[:, :self.c.ndim_x], x[:, :self.c.ndim_x])
            loss_backward += self.loss_weights.wasserstein_distance_weight * self.approx_wasserstein(x, output_rev)

        if self.prior:
            if self.dataset.lower() == 'ik_robotics':
                if self.prior_type == 'normal':
                    loss_backward += self.prior_level * normal_prior_x_loss(x_gt=x[:, :self.c.ndim_x], x=output_rev[:, :self.c.ndim_x])
                else:
                    loss_backward += self.prior_level * uniform_prior_loss(output_rev[:, :self.c.ndim_x], low=-0.5, high=0.5)

            elif self.dataset.lower() == 'gi':
                output_scaled = output_rev[:, :self.c.ndim_x] * self.x_std + self.x_mean
                loss_backward += config.prior_scale * uniform_prior_loss(output_scaled[:, 0], 200.5, 236.5)
                self.log("uniform_prior_loss_depth", uniform_prior_loss(output_scaled[:, 0], 200.5, 236.5))
                loss_backward += config.prior_scale * uniform_prior_loss(output_scaled[:, 1], 1532, 1592)
                self.log("uniform_prior_loss_speed", uniform_prior_loss(output_scaled[:, 1], 1532, 1592))

        l_tot = loss_backward + loss
        self.i_epoch += 1
        return l_tot

    def val_dataloader(self):
        if self.dataset.lower() == 'ik_robotics':
            _, val_loader = get_loaders()
            
        
        elif self.dataset.lower() == 'gi':
            _, val_loader, self.x_mean, self.x_std, self.y_mean, self.y_std = get_GI_loaders(param=self.param)
            self.x_mean = self.x_mean.to(config.device)
            self.x_std = self.x_std.to(config.device)
            self.y_mean = self.y_mean.to(config.device)
            self.y_std = self.y_std.to(config.device)
            
        return val_loader

    def configure_optimizers(self):
        
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr, betas=config.OptimizerConfig.betas, eps=config.OptimizerConfig.eps, weight_decay=config.OptimizerConfig.l2_reg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
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

        if self.dataset.lower() == 'ik_robotics':
            train_loader, _ = get_loaders()

        elif self.dataset.lower() == 'gi':
            train_loader,_, self.x_mean, self.x_std, self.y_mean, self.y_std = get_GI_loaders()
            self.x_mean = self.x_mean.to(config.device)
            self.x_std = self.x_std.to(config.device)
            self.y_mean = self.y_mean.to(config.device)
            self.y_std = self.y_std.to(config.device)
            
        
        
        return train_loader

    
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from loader import get_loaders, get_test, get_GI_loaders
from loss_functions import VLOSS, QLOSS, normal_prior_x_loss
from networks import get_inn, get_iresnet
from robot_arm_2d import RobotArm2d


class Q(nn.Module):
    def __init__(self, ndim, hidden_units, num_layers, inn_type):
        super().__init__()
        
        if inn_type == 'glow':
            self.model = get_inn(ndim, hidden_units=hidden_units, num_layers=num_layers)
        else:
            self.model = get_iresnet(ndim, num_blocks=num_layers, hidden_size=hidden_units)
        

    def forward(self, y, z, rev=False):
        inp = torch.cat([y,z], dim = 1)
        out, _ = self.model(inp, rev=rev)
        return out


class V(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(V, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is scalar score
        )

    def forward(self, x,y):
        inp = torch.cat([x,y], dim = 1)
        return self.model(inp)

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv')!=-1 or classname.find('Linear')!=-1:
    nn.init.normal_(m.weight.data,0.0,0.02)
  elif classname.find('BatchNorm')!=-1:
    nn.init.normal_(m.weight.data,1.0,0.02)
    nn.init.constant_(m.bias.data,0)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) != 11:
        print("Usage: python train_fdiv.py <divergence> <Prior> <Dataset> <latent_dim> <hidden_units> <seed> <num_layers> <fwd> <save_dir_log> <network_type>")
        sys.exit(1)
    divergence = sys.argv[1]
    prior = sys.argv[2]
    dataset = sys.argv[3]
    latent_dim = int(sys.argv[4])
    hidden_units = int(sys.argv[5])
    seed = int(sys.argv[6])
    num_layers = int(sys.argv[7])
    fwd = bool(int(sys.argv[8]))
    save_dir_log = sys.argv[9]
    network_type = sys.argv[10]
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    Q_criterion = QLOSS(divergence)
    V_criterion = VLOSS(divergence)
    loss_weights = config.LossWeightsConfig()

    if dataset == 'ik_robotics':
        c = config.IKConfig()
        ndim_x = c.ndim_x
        ndim_y = c.ndim_y
        ndim_tot = ndim_y + latent_dim
        ndim_z = latent_dim
        print("Total dim:", ndim_tot)
        Q_net = Q(ndim=ndim_y + ndim_z, hidden_units=hidden_units, num_layers=num_layers, inn_type=network_type).to(device)
        if not fwd:
            V_net = V(input_dim=ndim_x + ndim_y).to(device)
        else:
            V_net=V(input_dim=ndim_tot).to(device)
        train_loader, _  = get_loaders()
        
    elif dataset.lower() == 'gi':
        c = config.GIConfig()
        train_loader, test_loader, x_train_mean, x_train_std, y_train_mean, y_train_std  = get_GI_loaders()
        ndim_x = c.param_dim
        ndim_y = c.ndim_y
        ndim_tot = ndim_y + latent_dim
        ndim_z = latent_dim
        Q_net = Q(ndim=ndim_y + ndim_z, hidden_units=hidden_units, num_layers=num_layers, inn_type=network_type).to(device)
        if not fwd:
            V_net = V(input_dim=ndim_x + ndim_y).to(device)
        else:
            V_net=V(input_dim=ndim_tot).to(device)
        log_after_epochs = [0, 4, 9]
        true_posterior = torch.load('x_star_tensor.pt').detach().cpu().numpy()
        true_posterior = true_posterior[:2]
        depth, speed = true_posterior
        
        
    else:
        pass
        
        
    
    Q_optimizer = optim.Adam(Q_net.parameters(),lr=2e-4,betas=(0.5,0.9))
    V_optimizer = optim.Adam(V_net.parameters(),lr=2e-4,betas=(0.5,0.9))

    Q_net.apply(weights_init)
    V_net.apply(weights_init)
    Q_losses = []
    V_losses = []

    start = time.time()
    for ep in range(config.TrainingConfig.n_epochs):
        for i, (x, y) in enumerate(train_loader):
            b_size=x.shape[0]
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            #Train V
            V_net.zero_grad()
            x = x[:,:ndim_x]
            z = torch.randn(b_size,ndim_z).to(device) 
            yz = torch.cat([y, z], dim=1)
            if not fwd:
                v = V_net(x,y)
            else:
                v = V_net(y,z)
            
            v = torch.clamp(v, min=-4.0, max=4.0)
            loss_real = -V_criterion(v)
            loss_real.backward(retain_graph=True)

            if not fwd:
                fake_data = Q_net(y, z, rev = True)[:,:ndim_x]
            
            x_pad = torch.zeros(b_size, ndim_tot - ndim_x).to(device)
            
            if fwd:
                fake_data = Q_net(x, x_pad,rev=False)
            
            if not fwd:
                v_fake = V_net(fake_data.detach(), y)
            else:
                v_fake = V_net(fake_data.detach(), torch.tensor([]).to(device))

            
            loss_fake = -Q_criterion(v_fake)
            loss_fake.backward()#maximize F

            loss_V = -(loss_real + loss_fake)
            V_optimizer.step()

            #Train G 
            Q_net.zero_grad()
            if not fwd:
                v_fake = V_net(fake_data, y)
            else:
                v_fake = V_net(fake_data, torch.tensor([]).to(device))

            #loss_Q = Q_criterion(v_fake)# minimize F
            loss_Q = -V_criterion(v_fake)# maximize F (trick way)
            if prior == 'True':
                loss_Q += config.prior_scale * normal_prior_x_loss(x_gt=x[:, :ndim_x], x=fake_data[:, :ndim_x])
            if not fwd:
                loss_Q += torch.mean((fake_data - x) ** 2) * loss_weights.fdiv_reconstruction_weight  # L2 loss term
            if fwd:
                loss_Q += torch.mean((fake_data - yz) ** 2) * loss_weights.fdiv_reconstruction_weight  # L2 loss term for forward pass
            loss_Q.backward()
            Q_optimizer.step()


            if ep%1 == 0 and i ==0:
                print('Epoch [{}/{}], V_loss: {:.4f}, Q_loss: {:.4f}, T(x): {:.4f}, T(Q(z)): {:.4f}' 
                        .format(ep, config.TrainingConfig.n_epochs, loss_V.item(), loss_Q.item(), 
                                loss_real.mean().item(), loss_fake.mean().item()))
                Q_losses.append(loss_Q.item())
                V_losses.append(loss_V.item())
        if dataset.lower() == 'gi':   
            if ep in log_after_epochs:
                now = time.time()
                with torch.no_grad():
                    generated_x = []
                    z = torch.randn(config.num_samples, ndim_z).to(device)
                    y_star = torch.load('y_star_tensor.pt')
                    y_star = y_star.repeat(config.num_samples, 1).to(device)
                    x_gen = Q_net(y_star, z, rev = True)[:, :c.param_dim]
                    x_gen = x_gen.to('cpu').detach()
                    # Unnormalize the generated samples
                    x_gen[:, 0] = x_gen[:, 0] * x_train_std[0] + x_train_mean[0]
                    x_gen[:, 1] = x_gen[:, 1] * x_train_std[1] + x_train_mean[1]
                    depths = x_gen[:,0]
                    speeds = x_gen[:,1]
                    mse = torch.mean((x_gen[:,:c.param_dim] - true_posterior) ** 2).item()
                    # Log and save results
                    # print(f"GI MSE: {mse}")
                    mse_depth = torch.mean((depths - depth) ** 2).item()
                    mse_speed = torch.mean((speeds - speed) ** 2).item()
                    fdiv_info = {
                        "divergence": divergence,
                        "mse_depth": mse_depth,
                        "mse_speed": mse_speed,
                        "latent_dim": latent_dim,
                        "num_layers": num_layers,
                        "hidden_units": hidden_units,
                        "epoch": ep,
                        "seed": seed,
                        'time_elapsed': now - start
                    }
                    with open(f"{save_dir_log}.log", "a") as f:
                        f.write(json.dumps(fdiv_info) + "\n")
            
    
    end = time.time()
    
    if dataset == 'ik_robotics':
        robot = RobotArm2d(lengths=[0.5, 0.5, 1], sigmas=[0.25, 0.5, 0.5, 0.5])

        model = Q_net.to(device)
        y_clean = get_test()
        generated_x = []
        z = torch.randn(config.num_samples, latent_dim).to(device)
        y = y_clean.repeat([config.num_samples, 1]).to(device) #+ config.y_noise_scale * torch.randn(1, c.ndim_y).to(config.device)
            
        x_gen = model(y, z, rev = True)[:, :c.ndim_x]
        x_gen = x_gen.to('cpu').detach()

        #torch.save(x_gen, f"{divergence}_gen_lowz.pt")


        y_clean = y_clean.repeat(config.num_samples, 1)
        fig_name = f"{divergence}_prior" if prior == 'True' else f"{divergence}"
        print(fig_name)
        axes, distance,std = robot.viz_inverse(pos=y_clean, thetas=x_gen, save=False, show=False, fig_name=fig_name)
        log = {
            "latent_dim": latent_dim,
            "divergence": divergence,
            "distance": distance,
            "std": std,
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            "seed": seed,
            "time": (end - start)/60
        }
        with open(f"{save_dir_log}.log", "a") as f:
            f.write(json.dumps(log) + "\n")

    elif dataset.lower() == 'gi':
        model = Q_net.to(device)
        true_posterior = torch.load('x_star.pt').detach().cpu().numpy()
        true_posterior = true_posterior[:2]
        depth, speed = true_posterior
        generated_x = []
        z = torch.randn(config.num_samples, ndim_z).to(device)
        y_star = torch.load('y_star_tensor.pt')
        y_star = y_star.repeat(config.num_samples, 1).to(device)
        x_gen = model(y_star, z, rev = True)[:, :c.param_dim]
        x_gen = x_gen.to('cpu').detach()
        # Unnormalize the generated samples
        x_gen[:, 0] = x_gen[:, 0] * x_train_std[0] + x_train_mean[0]
        x_gen[:, 1] = x_gen[:, 1] * x_train_std[1] + x_train_mean[1]
        depths = x_gen[:,0]
        speeds = x_gen[:,1]
        mse = torch.mean((x_gen[:,:c.param_dim] - true_posterior) ** 2).item()
        # Log and save results
        # print(f"GI MSE: {mse}")
        mse_depth = torch.mean((depths - depth) ** 2).item()
        mse_speed = torch.mean((speeds - speed) ** 2).item()
        num_layers_val = sum(1 for n in getattr(Q_net.model, 'nodes', []) if 'coupling' in getattr(n, 'name', ''))
        # fdiv_info = {
        #     "divergence": divergence,
        #     "mse_depth": mse_depth,
        #     "mse_speed": mse_speed,
        #     "latent_dim": latent_dim,
        #     "num_layers": num_layers_val,
        #     "hidden_units": hidden_units,
        #     'training_time': end - start,
        #     "seed": seed
        # }
        # with open("gi_mse_fdiv.log", "a") as f:
        #     f.write(json.dumps(fdiv_info) + "\n")
        
        plt.figure(figsize=(2.5, 2.5))
        plt.hist(depths, color='darkblue', weights=np.ones_like(depths)/len(depths), edgecolor='black')
        plt.axvline(x=depth, color='red', ls='--', lw=1)#, label='True Value')

        plt.xlabel('Water Depth')
        plt.ylabel('Probability Density')
        plt.xlim([199, 237])
        name = f"{divergence}_water_depth_worst" # name of the plot file
        plt.savefig(name + ".png", bbox_inches="tight", dpi=300)
        plt.close()

        plt.figure(figsize=(2.5, 2.5))
        plt.hist(speeds, color='darkblue', weights=np.ones_like(speeds)/len(speeds), edgecolor='black')
        plt.axvline(x=speed, color='red', ls='--', lw=1) #, label='Truth = 1572')

        plt.xlabel('Sound Speed')
        plt.ylabel('Probability Density')
        plt.xlim([1492, 1592])
        # plt.legend()

        name = f"{divergence}_sound_speed_worst" # for saving the figure
        plt.savefig(name + ".png", bbox_inches="tight", dpi=300)
        plt.close()
        #torch.save(x_gen, f"{divergence}_gen_lowz.pt")


    else:
        
        pass

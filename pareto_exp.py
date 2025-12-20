import json
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from geomloss import SamplesLoss
from torch.utils.data import TensorDataset, DataLoader

from loss_functions import Activation_g, Conjugate_f, VLOSS, QLOSS
from networks import get_inn


def make_1d_pareto_dataset(shape, n_samples=60000, scale=1.0):
    data = (torch.from_numpy((np.random.pareto(shape, n_samples) + 1) * scale)).unsqueeze(1).to(torch.float32)
    labels = torch.zeros(n_samples)  # dummy labels for compatibility

    return TensorDataset(data, labels)


class Q(nn.Module):
    def __init__(self, ndim, hidden_units, num_layers):
        super().__init__()
        self.model = get_inn(ndim, hidden_units=hidden_units, num_layers=num_layers)

    def forward(self, z, rev=False):
        out, _ = self.model(z, rev=rev)
        return out


class V(nn.Module):
    def __init__(self, hidden_dim=64, data_dim=1):
        super(V, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is scalar score
        )

    def forward(self, x):
        return self.model(x)


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv')!=-1 or classname.find('Linear')!=-1:
    nn.init.normal_(m.weight.data,0.0,0.02)
  elif classname.find('BatchNorm')!=-1:
    nn.init.normal_(m.weight.data,1.0,0.02)
    nn.init.constant_(m.bias.data,0)

def wasserstein1(x,y):
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01)
    return loss(x, y)

if __name__ == "__main__":
    
    divergence = "KLD" # GAN, JSD, RKL, KLD 
    modelName = "f-GAN-"+ divergence
    batch_size = 512
    workers = 2
    epochs = 20
    data_dim = 1
    latent_dim = 10
    num_layers = 6
    c_dim = 1
    hidden_units = 64
    TINY = 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(sys.argv) != 3:
        print("Usage: pareto_exp.py <seed> <alpha>")
        sys.exit(1)
    manualSeed = int(sys.argv[1])
    alpha = float(sys.argv[2])

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    print("alpha:", alpha)
    train_set = make_1d_pareto_dataset(shape=alpha)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    Q_net = Q(ndim=latent_dim, hidden_units=hidden_units, num_layers=num_layers).to(device)
    V_net = V(hidden_dim=64, data_dim=data_dim).to(device)

    Q_criterion =QLOSS(divergence)
    V_criterion =VLOSS(divergence)

    Q_optimizer = optim.Adam(Q_net.parameters(),lr=2e-4,betas=(0.5,0.9))
    V_optimizer = optim.Adam(V_net.parameters(),lr=2e-4,betas=(0.5,0.9))

    fixed_noise = torch.randn(batch_size, latent_dim,device=device)
    Q_net.apply(weights_init)
    V_net.apply(weights_init)
    Q_losses = []
    V_losses = []

    iter_per_plot = 250
    plot_per_eps=(int(len(train_loader)/iter_per_plot)+1)

    
    for ep in range(epochs):
        for i, (data, _) in enumerate(train_loader):
            b_size=data.shape[0]
            data = data.to(torch.float32).to(device)
            #Train V
            V_net.zero_grad()
            try:
                v = V_net(data)

            except:
                import pdb; pdb.set_trace()
            v = torch.clamp(v, min=-4.0, max=4.0)

            #import pdb; pdb.set_trace()  
            loss_real = -V_criterion(v)
            loss_real.backward(retain_graph=True)

            z = torch.randn(b_size,latent_dim).to(device) 
            fake_data = Q_net(z, rev = True)[:,:data_dim].unsqueeze(1)
            v_fake = V_net(fake_data.detach())
            v_fake = torch.clamp(v_fake, min=-4.0, max=4.0)

            
            loss_fake = -Q_criterion(v_fake)
            loss_fake.backward()#maximize F

            loss_V = -(loss_real + loss_fake)
            V_optimizer.step()

            #Train G 
            Q_net.zero_grad()
            v_fake = V_net(fake_data)

            #loss_Q = Q_criterion(v_fake)# minimize F
            loss_Q = -V_criterion(v_fake)# maximize F (trick way)
            loss_Q.backward()
            Q_optimizer.step()


            if (i+1)%iter_per_plot == 0 or i ==0:
                print('Epoch [{}/{}], Step [{}/{}], V_loss: {:.4f}, Q_loss: {:.4f}, T(x): {:.4f}, T(Q(z)): {:.4f}' 
                        .format(ep, epochs, i+1, len(train_loader), loss_V.item(), loss_Q.item(), 
                                loss_real.mean().item(), loss_fake.mean().item()))
                Q_losses.append(loss_Q.item())
                V_losses.append(loss_V.item())
                
                
    # Generate latent noise
    z = torch.randn(10000, latent_dim).to(device) 
    # Generate samples using Q
    with torch.no_grad():
        samples = Q_net(z, rev = True)[:,:data_dim] # Q_net is your trained generator

    # Move to CPU and convert to numpy if needed
    samples_np = samples.cpu().numpy()

    
    ws = wasserstein1(samples, train_set[:10000][0].to(device))
    
    # prepare scalar values
    if isinstance(ws, torch.Tensor):
        ws_val = ws.detach().cpu().item()
    else:
        ws_val = float(ws)

    metrics = {"alpha": float(alpha), "ws": ws_val}
    print("Metrics:", metrics)
    
    # optionally persist to file
    with open("observation_6_moments.log", "a") as f:
        json.dump(metrics, f)
        f.write("\n")



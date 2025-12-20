import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from scipy.stats import norm

import config
from loader import get_test, get_GI_loaders
from models import INNModel
from robot_arm_2d import RobotArm2d

entropy_reg = float(os.getenv("OPTUNA_ENTROPY_REG", 0.01))

    


def set_seed(seed: int = 42):
    """Seed all randomness sources for reproducibility."""
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

        
        

class GILoggingCallback(pl.callbacks.Callback):
    """
    PyTorch Lightning callback to run GI posterior sampling and MSE logging
    after epochs 1, 5, and 10.
    """

    def __init__(self, loss_f, prior, latent_dim,
                hidden_units, num_layers, seed, start):
        super().__init__()
        self.loss_f = loss_f
        self.prior = prior
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.seed = seed
        self.start = start
        # Epochs on which to trigger logging
        self.target_epochs = {0, 4, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99}  # 0-based indexing for epochs 1, 5, and 10

    def on_train_epoch_end(self, trainer, pl_module):
        num_epochs = trainer.current_epoch
        if num_epochs not in self.target_epochs:
            return  # skip other epochs

        model = pl_module
        now = time.time()

        print(f"Running GI logging routine after epoch {num_epochs}...")

        # Start time tracking
        paths = config.PathsConfig()

        # Load reference tensors
        x_star = torch.load(paths.x_star_file).to(config.device)
        y_star = torch.load(paths.y_star_file).to(config.device).unsqueeze(0)

        # Load normalization stats
        _, _, x_mean, x_std, y_mean, y_std = get_GI_loaders()

        eps = 1e-8
        # Convert means/stds to tensors
        y_mean_t = torch.as_tensor(y_mean, device=config.device, dtype=y_star.dtype)
        y_std_t = torch.as_tensor(y_std, device=config.device, dtype=y_star.dtype)

        # Normalize target y
        y_star = (y_star - y_mean_t) / (y_std_t + eps)

        # Instantiate GI config
        c = config.GIConfig()

        # Move model to device
        device = config.device
        model = model.to(device)
        num_samples = 1000
        y_batch = y_star.expand(num_samples, -1).to(device)
        z_batch = torch.randn(num_samples, self.latent_dim, device=device)
        inputs = torch.cat([y_batch, z_batch], dim=1)

        with torch.no_grad():
            outputs = model(inputs, rev=True)[0]

        # Unnormalize generated samples
        x_mean_t = torch.as_tensor(x_mean, device=outputs.device, dtype=outputs.dtype)
        x_std_t = torch.as_tensor(x_std, device=outputs.device, dtype=outputs.dtype)

        outputs[:, :c.param_dim] = outputs[:, :c.param_dim] * x_std_t[:c.param_dim] + x_mean_t[:c.param_dim]
        posterior_samples = outputs.detach().cpu()

        # Reference
        x_ref = x_star.detach().cpu()
        if x_ref.dim() == 1:
            x_ref = x_ref.unsqueeze(0)

        ps = posterior_samples[:, :c.param_dim]
        x_exp = x_ref.expand(ps.size(0), -1)[:, :c.param_dim]

        # Compute MSE
        mse = torch.mean((ps - x_exp) ** 2).item()

        # Compute componentwise MSEs
        depths = ps[:, 0].numpy()
        speeds = ps[:, 1].numpy()
        depth, speed = x_star[0].cpu().numpy(), x_star[1].cpu().numpy()
        mse_depth = float(np.mean((depths - depth) ** 2))
        mse_speed = float(np.mean((speeds - speed) ** 2))

        
        #print(f"[Epoch {num_epochs}] GI MSE depth: {mse_depth:.6f}, speed: {mse_speed:.6f}")

        # Log entry
        log_entry = {
            "loss": self.loss_f,
            "prior": self.prior,
            "latent_dim": self.latent_dim,
            "mse_depth": mse_depth,
           "mse_speed": mse_speed,
            "training_time": float(now - self.start),
            "num_epochs": num_epochs,
            "seed": self.seed,
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers
        }

        with open(paths.gi_mse_nll_log, "a") as f:
            f.write(f"{log_entry}\n")


if __name__ == '__main__':
    
    
    
    if len(sys.argv) != 14:
        print("Usage: train.py <loss_function> <dataset> <num_layers> <num_epochs> <latent_dim> <seed> <hidden_units> <learning_rate> <network_type> <prior_level> <prior_type> <save_dir> <entropy_reg>")
        sys.exit(1)
    loss_f = sys.argv[1]
    dataset = sys.argv[2]
    num_layers = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    latent_dim = int(sys.argv[5])
    seed = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    network_type = sys.argv[9]
    prior_level = float(sys.argv[10])
    prior_type = sys.argv[11]
    save_dir = sys.argv[12]
    entropy_reg = float(sys.argv[13])
    prior = True if prior_level > 0 else False
    set_seed(seed)
    print(f"Loss function: {loss_f}, Prior: {prior}, Dataset: {dataset}, num_layers: {num_layers}, learning_rate: {learning_rate} and latent_dim: {latent_dim}, seed: {seed}, hidden_units: {hidden_units}, network_type: {network_type}, prior_level: {prior_level}, prior_type: {prior_type}  ")
    model = INNModel(prior=prior, loss_f=loss_f, dataset=dataset, hidden_units=hidden_units, entropy_reg=entropy_reg, num_layers=num_layers, latent_dim=latent_dim, lr=learning_rate, network_type=network_type, prior_level=prior_level, prior_type=prior_type)

    

    if dataset.lower() == 'gi':

        paths = config.PathsConfig()
        # Load the fixed point data for GI
        x_star = torch.load(paths.x_star_file)
        y_star = torch.load(paths.y_star_file)
        c = config.GIConfig()
        
        start = time.time()
        logging_callback = GILoggingCallback(
            loss_f=loss_f,
            prior=prior,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            num_layers=num_layers,
            seed=seed,
            start = start
        )
        
        trainer = pl.Trainer(
            max_epochs=num_epochs, 
            #callbacks=[logging_callback],
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm'          # Log every step for better tracking
        )
    
    
    else:
        trainer = pl.Trainer(max_epochs=num_epochs)
    
    
    start = time.time()
    trainer.fit(model)
    end = time.time()
    print(f"Training time: {end-start}")
    paths = config.PathsConfig()
    log_entry = {
        "training_time": float(end - start),

    }

    with open(paths.training_time_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    if dataset.lower() == 'ik_robotics':

        c = config.IKConfig()
        arm = RobotArm2d(lengths=c.lengths, sigmas=c.sigmas, viz_dir=save_dir)
        model = model.to(config.device)
        y_clean = get_test()
        ndim_tot = c.ndim_y + latent_dim
        y_test = y_clean
        generated_x = []
        for _ in range(config.num_samples):
            z = torch.randn(1, latent_dim).to(config.device)
            pad_yz = torch.zeros(1, ndim_tot - c.ndim_y - latent_dim).to(config.device)
            y_noisy = y_test.unsqueeze(0).to(config.device)
            y_aug = torch.cat([z, pad_yz, y_noisy], dim=1)
            x_gen = model(y_aug, rev=True)[0]
            generated_x.append(x_gen)

        gen_x = torch.stack(generated_x).squeeze(1)[:, :c.ndim_x]
        gen_x = gen_x.to('cpu').detach()
        y_clean = y_clean.repeat(config.num_samples, 1)
        fig_name = f"{loss_f}_{prior_type}" if prior else loss_f
        axes, distance, std = arm.viz_inverse(pos=y_clean, thetas=gen_x, save=True, show=False, fig_name=fig_name)
        print(f"RMSE distance: {distance}, std: {std}")
        log_data = {
            "loss_function": loss_f,
            "prior": prior,
            "latent_dim": latent_dim,   
            "fig_name": fig_name,
            "distance": float(distance) if hasattr(distance, '__float__') else distance,
            "std": float(std) if hasattr(std, '__float__') else std,
            "training_time": float(end-start) 
        }
        with open(f"{save_dir}.log", "a") as f:
            f.write(f"{log_data}\n")

        

    elif dataset.lower() == 'gi':

        x_star = torch.load(paths.x_star_file).to(config.device)
        y_star = torch.load(paths.y_star_file).to(config.device).unsqueeze(0)

        # Load normalization stats
        _, _, x_mean, x_std, y_mean, y_std = get_GI_loaders()

        eps = 1e-8
        # Convert means/stds to tensors
        y_mean_t = torch.as_tensor(y_mean, device=config.device, dtype=y_star.dtype)
        y_std_t = torch.as_tensor(y_std, device=config.device, dtype=y_star.dtype)

        # Normalize target y
        y_star = (y_star - y_mean_t) / (y_std_t + eps)

        # Instantiate GI config
        
        c = config.GIConfig()
        
        # Move model to device
        device = config.device
        model = model.to(device)
        num_samples = 1000
        y_batch = y_star.expand(num_samples, -1).to(device)
        z_batch = torch.randn(num_samples, latent_dim, device=device)
        inputs = torch.cat([y_batch, z_batch], dim=1)

        with torch.no_grad():
            outputs = model(inputs, rev=True)[0]

        # Unnormalize generated samples
        x_mean_t = torch.as_tensor(x_mean, device=outputs.device, dtype=outputs.dtype)
        x_std_t = torch.as_tensor(x_std, device=outputs.device, dtype=outputs.dtype)

        outputs[:, :c.param_dim] = outputs[:, :c.param_dim] * x_std_t[:c.param_dim] + x_mean_t[:c.param_dim]
        posterior_samples = outputs.detach().cpu()

        # Reference
        x_ref = x_star.detach().cpu()
        if x_ref.dim() == 1:
            x_ref = x_ref.unsqueeze(0)

        ps = posterior_samples[:, :c.param_dim]
        x_exp = x_ref.expand(ps.size(0), -1)[:, :c.param_dim]

        # Compute MSE
        mse = torch.mean((ps - x_exp) ** 2).item()
        depths = ps[:, 0].numpy()
        speeds = ps[:, 1].numpy()
        depth, speed = x_star[0].cpu().numpy(), x_star[1].cpu().numpy()
    
        plt.figure(figsize=(2.5, 2.5))
        plt.hist(depths, color='darkblue', weights=np.ones_like(depths)/len(depths), edgecolor='black')
        plt.axvline(x=depth, color='red', ls='--', lw=1)

        plt.xlabel('Water Depth')
        plt.ylabel('Probability Density')
        plt.xlim([199, 237])
        name = f"{loss_f}_depth"
        plt.savefig(name + ".png", bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(2.5, 2.5))
        plt.hist(speeds, color='darkblue', weights=np.ones_like(speeds)/len(speeds), edgecolor='black')
        plt.axvline(x=speed, color='red', ls='--', lw=1)

        plt.xlabel('Sound Speed')
        plt.ylabel('Probability Density')
        plt.xlim([1492, 1592])
        name = f"{loss_f}_speed"
        plt.savefig(name + ".png", bbox_inches='tight', dpi=300)
        plt.close()
    

        print(json.dumps({"val_loss": mse}))

        
    


    

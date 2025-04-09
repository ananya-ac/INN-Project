import torch
import config
from time import time
from models import INNModel
from tqdm import tqdm
from loader import get_test
import pytorch_lightning as pl
import pdb
from robot_arm_2d import RobotArm2d
import time
from lightning.pytorch.loggers import WandbLogger
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from loader import get_GI_loaders
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
import torch
import numpy as np
from typing import Any, Dict, Optional
import os
import pandas as pd


class MinValLossFixedPointEvaluationCallback(pl.callbacks.Callback):
    def __init__(
        self, 
        y_star: torch.Tensor, 
        verbose: bool = True,
        prior:bool = False,
        loss_f:str = 'NLL',
        ndim_z: int = None
    ):
        """
        Callback that runs evaluation on a specific fixed point when val_loss reaches its minimum.
        
        Args:
            fixed_point_data: The fixed point data to evaluate on (single sample or batch)
            fixed_point_target: Optional target for the fixed point if computing a supervised metric
            metric_name: Name of the metric to log
            verbose: Whether to print evaluation results
        """
        super().__init__()
        self.y_star = y_star.unsqueeze(0)
        self.verbose = verbose
        self.best_val_loss = float('inf')
        self.prior = prior
        self.loss_f = loss_f
        self.best_model_state = None
        self.ndim_z = ndim_z
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """After validation, check if we have a new minimum validation loss."""
        current_val_loss = trainer.callback_metrics.get("val_loss")
        
        if current_val_loss is None:
            return
        
        # Convert to float in case it's a tensor
        current_val_loss = float(current_val_loss)
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            
            # Evaluate on the fixed point
            self._evaluate_fixed_point(pl_module, trainer)
    
    def _evaluate_fixed_point(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        """Evaluate the model on the fixed point data."""
        # Set model to eval mode
        pl_module.eval()
        device = pl_module.device
        with torch.no_grad():
            # Handle both supervised and unsupervised cases
            posterior_samples = [pl_module.sample(torch.cat([self.y_star.to(device), torch.randn(self.y_star.shape[0], self.ndim_z).to(device)], dim=1)) for i in range(1000)]
            posterior_samples = [posterior_samples[i].detach().cpu() for i in range(len(posterior_samples))]
            torch.save(posterior_samples, f'posterior_samples_{self.prior}_{self.loss_f}.pt')
            print(f"Samples saved to posterior_samples_{self.prior}_{self.loss_f}.pt at {trainer.current_epoch}")
            
        pl_module.train()



if __name__ == '__main__':
    
    
    if len(sys.argv) != 4:
        print("Usage: train.py <loss_function> <prior> <dataset>")
        sys.exit(1)
    torch.autograd.set_detect_anomaly(True)
    loss_f = sys.argv[1]
    prior = sys.argv[2].lower() == 'true'
    dataset = sys.argv[3]
    print(f"Loss function: {loss_f}, Prior: {prior}, Dataset: {dataset}")
    model = INNModel(prior=prior, loss_f=loss_f, dataset=dataset)
    wandb_logger = WandbLogger(project="INN")
    wandb_logger.watch(model, log="all")
    if dataset.lower() == 'gi':
        # Load the fixed point data for GI
        x_star = torch.load('x_star_tensor.pt')
        y_star = torch.load('y_star_tensor.pt')
        c = config.GI_Config()
        
        # Create the callback
        min_val_loss_callback = MinValLossFixedPointEvaluationCallback(
            y_star=y_star,
            verbose=True,
            prior=prior,
            loss_f=loss_f,
            ndim_z=c.ndim_z 
        )
        
        trainer = pl.Trainer(
            max_epochs=config.n_epochs, 
            logger=wandb_logger,
            callbacks=[min_val_loss_callback],  # Add the custom callback
            log_every_n_steps=1,
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm'          # Log every step for better tracking
        )
    
    
    else:
        trainer = pl.Trainer(max_epochs=config.n_epochs, logger = wandb_logger)
    
    
    start = time.time()
    trainer.fit(model)
    end = time.time()
    print(f"Training time: {end-start}")
    wandb_logger.log_metrics({"training_time": end-start})
    
    
    if dataset.lower() == 'ik_robotics':
        c = config.IK_Config()
        arm = RobotArm2d(lengths = c.lengths, sigmas = c.sigmas)
        model = model.to(config.device)
        y_clean = get_test()
        y_test = y_clean
        #generated_x = [model(torch.cat([(y_test.unsqueeze(0)).to(config.device),  config.y_noise_scale * torch.randn(size = (1, c.ndim_tot - c.ndim_y)).to(config.device)], dim = 1), rev = True)[0] for i in range(config.num_samples)]
        generated_x = [model(torch.cat([(y_test.unsqueeze(0)).to(config.device) + config.y_noise_scale * torch.randn(1, c.ndim_y).to(config.device),  config.y_noise_scale * torch.randn(size = (1, c.ndim_tot - c.ndim_y)).to(config.device)], dim = 1), rev = True)[0] for i in range(config.num_samples)]
        gen_x = torch.stack(generated_x).squeeze(1)[:, :c.ndim_x]
        gen_x = gen_x.to('cpu')
        gen_x = gen_x.detach()
        y_clean = y_clean.repeat(config.num_samples,1)
        
        fig_name = f"{loss_f}_prior" if prior else loss_f 
        axes, distance = arm.viz_inverse(pos = y_clean, thetas = gen_x, save = False, show = False, fig_name=fig_name)
        csv_path = 'rmse_logs.csv'

        if not os.path.exists(csv_path):
            # Create a new DataFrame with the column and add the value
            df = pd.DataFrame({f"{loss_f}_{prior}": [distance]})
            df.to_csv(csv_path, index=False)
        else:
            # Load the existing CSV and add the value to the column
            df = pd.read_csv(csv_path)
            column_name = f"{loss_f}_{prior}"
            if column_name not in df.columns:
                df[column_name] = None  # Add the column if it doesn't exist
            df = pd.concat([df, pd.DataFrame({column_name: [distance]})], ignore_index=True)
            df.to_csv(csv_path, index=False)
        # with open(f'RMSE_logs_{loss_f}_{prior}.txt', 'a') as file: 
        #     file.write(str(distance))
        #     file.write("\n")
            
        
    elif dataset.lower() == 'cubic':
        y_clean = torch.tensor(-2.5**3, dtype = torch.float)
        c = config.Cubic_Config()
        model = model.to(config.device)
        gen_x = [model(torch.cat([(y_clean + config.y_noise_scale * torch.randn_like(y_clean)).view(1,-1).to(config.device), config.y_noise_scale *  torch.randn(size = (1,c.ndim_z)).to(config.device)], dim = 1), rev = True)[0] for i in range(config.num_samples)]
        gen_x = torch.stack(gen_x).squeeze(1)[:, :c.ndim_x]
        gen_x = gen_x.to('cpu')
        gen_x = gen_x.detach().numpy()
        print('mean of samples:', gen_x.mean())
        plt.rcParams['font.family'] = 'DejaVu Serif'
        sns.set_theme(style="whitegrid", palette="muted")
        plt.figure(figsize=(10, 6))
        sns.histplot(gen_x, bins=55,kde = True, color="skyblue", linewidth=2, stat="probability", alpha = 1, multiple='dodge', hatch='\\\\', label = 'No Prior' if not prior else 'Prior')
        plt.xlabel("Value", fontsize=20, fontname = 'DejaVu Serif')
        plt.ylabel("Density", fontsize=20, fontname = 'DejaVu Serif')
        plt.legend(prop = 'DejaVu Serif')
        sns.despine()
        plt.grid(True, linestyle=' ', color='gray', alpha=0.6)
        plt.show()
        
    elif dataset.lower() == 'gi':
        
        # x_star = torch.load('x_star.pt').to(config.device)
        # y_star = torch.load('y_star.pt').to(config.device).unsqueeze(0)
        # c = config.GI_Config()
        # model = model.to(config.device)
        # posterior_samples = [model(torch.cat([y_star, torch.randn(y_star.shape[0], c.ndim_z).to(config.device)], dim=1), rev = True)[0] for i in range(1000)]
        # posterior_samples = [posterior_samples[i].detach().cpu() for i in range(len(posterior_samples))]
        # torch.save(posterior_samples, f'posterior_samples_{prior}_{loss_f}.pt')
        print("Training completed for GI dataset.")
            
    
    else:
        print("Invalid dataset. Please choose from 'ik_robotics', 'cubic', or 'gi'.")
        sys.exit(1)
    
        
        
        
        


    


    

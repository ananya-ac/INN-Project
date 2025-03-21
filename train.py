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
import torch.functional as F
import numpy as np




if __name__ == '__main__':
    
    
    if len(sys.argv) != 4:
        print("Usage: train.py <loss_function> <prior> <dataset>")
        sys.exit(1)

    loss_f = sys.argv[1]
    prior = sys.argv[2].lower() == 'true'
    dataset = sys.argv[3]

    # prior = False
    # loss_f = 'NLL'
    model = INNModel(prior=prior, loss_f=loss_f, dataset=dataset)
    wandb_logger = WandbLogger(project="INN")
    wandb_logger.watch(model, log="all")
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
        with open('RMSE_logs.txt', 'a') as file: 
            file.write(str(distance))
            file.write("\n")
            
        
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
        
        _, val_loader = get_GI_loaders()
        c = config.GI_Config()
            
        loss = 0
        
        for batch in val_loader:
            x,y = batch
            y = torch.cat([y, config.y_noise_scale * torch.randn(y.shape[0], c.ndim_z)], dim=1)
            y = y.to(config.device)
            pdb.set_trace()
            output_rev, _ = model(y, rev=True)
            loss += F.mse_loss(x[:, :c.ndim_x], output_rev[:, :c.ndim_x])
        
        print("val_loss", loss)
            
    
    else:
        print("Invalid dataset. Please choose from 'ik_robotics', 'cubic', or 'gi'.")
        sys.exit(1)
    
        
        
        
        


    


    

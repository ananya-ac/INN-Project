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
        
        #generated_x = [model(torch.cat([y_test.unsqueeze(0).to(c.device), torch.randn(size = (1, c.ndim_tot - c.ndim_y)).to(c.device)], dim = 1), rev = True)[0] for i in range(c.num_samples)]
        gen_x = torch.stack(generated_x).squeeze(1)[:, :c.ndim_x]
        gen_x = gen_x.to('cpu')
        gen_x = gen_x.detach()
        y_clean = y_clean.repeat(config.num_samples,1)
        
        fig_name = f"{loss_f}_prior" if prior else loss_f 
        axes, distance = arm.viz_inverse(pos = y_clean, thetas = gen_x, save = True, show = False, fig_name=fig_name) 
            
        
    
    else:
        print("Invalid dataset. Please choose 'ik_robotics'")
        sys.exit(1)
    
        
        
        
        


    


    

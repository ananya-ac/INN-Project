import torch

var = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
y_noise_scale = 1
zeros_noise_scale = 5e-2
prior_scale = 1
      

# Cubic Equation Dataset

class Cubic_Config():
    ndim_x = 1
    ndim_y = 1
    ndim_z = 1
    ndim_tot = 2
    

#IK Robotics Dataset

class IK_Config():
    ndim_x = 4
    ndim_y = 2
    ndim_z = 2
    ndim_tot = 16
    num_joints = 4
    lengths = [0.5, 0.5, 1] 
    sigmas = [0.25, 0.5, 0.5, 0.5]

#Geoacoustic Inversion Dataset    
class GI_Config():
    ndim_x = 72
    ndim_y = 70
    ndim_z = 2
    ndim_tot = 72
    param_dim = 2
    
#Toy Dataset
class Toy_Config():
    ndim_x = 1
    ndim_y = 1
    ndim_z = 1
    ndim_tot = 2
    


#DataLoader
batch_size = 2048
train_workers = 16


#Training 
n_epochs = 50 # Number of epochs for training
val_split = 0.8

#Optimizer
lr = 3e-4
l2_reg = 2e-5
betas = (0.8, 0.9)
eps = 1e-6

#model
num_layers = 4


#uniform_prior
lambda_ = 1.0

#generation
num_samples = 1000

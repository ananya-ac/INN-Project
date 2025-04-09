import pickle
import torch
import config as c
import numpy as np
from scipy.io import loadmat
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import pdb
def get_loaders():
    
    with open('./data/inverse_data_4_100_1000.pickle', 'rb') as file:
        data = pickle.load(file)
        
    x = data['thetas']
    y = data['pos']
    y = y.repeat_interleave(1000, dim=0)
    
    test_split = round(c.val_split * x.shape[0])
    
    test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x[test_split:], y[test_split:]),
    batch_size=c.batch_size, shuffle=True, drop_last=True)


    train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x[:test_split], y[:test_split]),
    batch_size=c.batch_size, shuffle=True, drop_last=True)
    
    return train_loader, test_loader

def get_test():
    
    with open('./data/inverse_data_4_100_1000.pickle', 'rb') as file:
        data = pickle.load(file)
        
    y = data['pos']
    #y_test = y[5]
    y_test = y[0]
    
    
    return y_test


def cubic_equation_loaders(num_samples):
    
    
    x = np.random.uniform(low = -2.5, high = 2.5, size = num_samples)
    y = x**3 
    test_split = round(c.val_split * num_samples)
    x_train, x_test = x[:test_split], x[test_split:]
    y_train, y_test = y[:test_split], y[test_split:]

    train_loader = DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=c.batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
            batch_size=c.batch_size, shuffle=False, drop_last=True)
    
    
    return train_loader, test_loader
    

def get_GI_loaders(data_dir = 'GI_Data/data2D.mat', num_receivers = 5):
        
        data = loadmat(data_dir)
        output_data = np.transpose(data['data'] + data['noise'], axes=(2,0,1)) #shape: (num_samples, num_receivers, freq)
        noise_data  = np.transpose(data['noise'], axes = (2,0,1)) #shape: (num_samples, num_receivers, freq)
        params = data['samples'].T #shape: (num_samples, num_params)
        real_part = np.real(output_data)  # Shape: (num_samples, num_receivers, freq)
        imag_part = np.imag(output_data)  # Shape: (num_samples, num_receivers, freq)
        noise_real_part = np.real(noise_data)  # Shape: (num_samples, num_receivers, freq)
        noise_imag_part = np.imag(noise_data)  # Shape: (num_samples, num_receivers, freq)
        output = np.concatenate([real_part,imag_part], axis = 2) # Shape: (num_samples, num_receivers, 2 x freq)
        noise = np.concatenate([noise_real_part, noise_imag_part], axis = 2) # Shape: (num_samples, num_receivers, 2 x freq)
        output = output[:,:num_receivers,:].reshape(100000,-1)
        noise = noise[:,:num_receivers,:].reshape(100000,-1)
        x = np.concatenate([params, noise], axis = 1)
        y = output[:]
        num_samples = x.shape[0]
        test_split = round(c.val_split * num_samples)
        x_train, x_test = x[:test_split], x[test_split:]
        y_train, y_test = y[:test_split], y[test_split:]
        x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
        y_train_mean, y_train_std = y_train.mean(axis=0), y_train.std(axis=0)
        x_train_mean, x_train_std = torch.tensor(x_train_mean, dtype=torch.float32), torch.tensor(x_train_std, dtype=torch.float32)
        y_train_mean, y_train_std = torch.tensor(y_train_mean, dtype=torch.float32), torch.tensor(y_train_std, dtype=torch.float32)
        
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=c.batch_size, shuffle=True, drop_last=True)

        test_loader = DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
                batch_size=c.batch_size, shuffle=False, drop_last=True)
    
        return train_loader, test_loader,x_train_mean, x_train_std, y_train_mean, y_train_std
    
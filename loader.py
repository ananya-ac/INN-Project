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
    

def get_GI_loaders():
        
        
        data_dir = 'GI_Data/One/data2DConstVar_10k.mat'
        data = loadmat(data_dir)
        output_data = np.transpose(data['data'] + data['noise'], axes=(0, 2, 1)) # + data['noise']
        noise_data  = np.transpose(data['noise'], axes=(0, 2, 1))
        params      = np.transpose(data['samples'])

        water_depths = params[:,0].reshape(-1, 1)
        speeds = params[:,1].reshape(-1, 1)

        n_receivers    = 1 #len(output_data)
        n_frequencies  = len(output_data.T)
        n_examples     = len(params)

        data_rows = []
        columns = ['Speed', 'Water_Depth'] #, Tilt, Thickness

        for example_idx in range(n_examples):
            row = []
            row.append(speeds[example_idx][0])
            row.append(water_depths[example_idx][0])

            for receiver_idx in range(n_receivers):

                #noise data
                for freq_idx in range(n_frequencies):
                    row.append(noise_data[receiver_idx, example_idx, freq_idx].real)
                    row.append(noise_data[receiver_idx, example_idx, freq_idx].imag)
                    if example_idx == 0:
                        columns += [f'R{receiver_idx+1}_Noise_{freq_idx+1}_Real', f'R{receiver_idx+1}_Noise_{freq_idx+1}_Imag']
                
                # output data
                for freq_idx in range(n_frequencies):
                    row.append(output_data[receiver_idx, example_idx, freq_idx].real)
                    row.append(output_data[receiver_idx, example_idx, freq_idx].imag)
                    if example_idx == 0:
                        columns += [f'R{receiver_idx+1}_Output_{freq_idx+1}_Real', f'R{receiver_idx+1}_Output_{freq_idx+1}_Imag']
                
            data_rows.append(row)

        df = pd.DataFrame(data_rows, columns=columns)
        data = df.copy()
        x  = data[['Speed', 'Water_Depth']]
        y = data.drop(columns=['Speed', 'Water_Depth'])
        x  = torch.tensor(x.values.astype(np.float32))
        y = torch.tensor(y.values.astype(np.float32))
        num_samples = x.shape[0]
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
    
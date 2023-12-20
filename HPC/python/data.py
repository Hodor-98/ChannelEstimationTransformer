#=======================================================================================================================
#=======================================================================================================================
import os
import math
import time
import numpy as np
import scipy.io as scio
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft 
pi = np.pi



def LoadBatch(H):
    '''
    H: T * M * Nr * Nt
    ''' 
    M, T, Nr, Nt = H.shape 
    H = H.reshape([M, T, Nr * Nt])
    H_real = np.zeros([M, T, Nr * Nt, 2])
    H_real[:,:,:,0] = H.real 
    H_real[:,:,:,1] = H.imag 
    H_real = H_real.reshape([M, T, Nr*Nt*2])
    H_real = torch.tensor(H_real, dtype = torch.float32)
    return H_real

def real2complex(data):
    B, P, N = data.shape 
    data2 = data.reshape([B, P, N//2, 2])
    data2 = data2[:,:,:,0] + 1j * data2[:,:,:,1]
    return data2

def noise(H, SNR):
    
    sigma = 10 ** (- SNR / 10) 
    # Generate complex Gaussian noise with PyTorch
    real_part = torch.randn(*H.shape)
    imag_part = torch.randn(*H.shape)
    noise = np.sqrt(sigma / 2) * (real_part + 1j * imag_part)

    # Normalize the noise
    noise = noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))
    

    return H + noise

def channelnorm(H):
    H = H / torch.sqrt(torch.mean(np.abs(H)**2))
    return H


class SeqData(Dataset):
    def __init__(self, dataset_name, seq_len, pred_len, SNR = 14):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length =seq_len+pred_len
        self.SNR = SNR
        
        with open(dataset_name, 'rb') as handle:
            self.dataset = pickle.load(handle)
            
    def __len__(self):
        return self.dataset.shape[0] 
    
    def __getitem__(self, idx):
        seed = math.floor(math.modf(time.time())[0]*500*320000)**2 % (2**32 - 2)
        np.random.seed(seed)
        
        H = self.dataset[idx]
        M,T,Nr,Nt = H.shape 
        L = self.length      
        
        start = np.random.randint(0, T-L+1) 
        end = start + L
        
        # H = channelnorm(H)
        
        # H = noise(H, self.SNR)
        
        H = H[:, start:end, ...]
        H_pred = H[:,self.seq_len:,...]
        H_seq = H[:,0:self.seq_len, ...]
        
        index = np.random.choice(M, M, replace = False)    
        H = H[index, ...]
        H_seq = H_seq[index, ...]
        H_pred = H_pred[index, ...] 
        
        return H, H_seq, H_pred


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


class SeqData(Dataset):
    def __init__(self, dataset_name, seq_len, pred_len, SNR = 14):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length =seq_len+pred_len
        
        with open(dataset_name, 'rb') as handle:
            self.dataset = pickle.load(handle)
        
    def __len__(self):
        return self.dataset.size[0]
    
    def __getitem__(self, idx):
        seed = math.floor(math.modf(time.time())[0]*500*320000)**2 % (2**32 - 2)
        np.random.seed(seed)
        
        print(self.dataset.shape)
        
        H = self.dataset[idx]

        M,T,Nr,Nt = H.shape  #slot数 * 子载波数 * 基站天线数 * 用户天线数
        L = self.length       #序列长度
        
        start = np.random.randint(0, T-L+1) #序列开始位置
        end = start + L
        
        H = H[:, start:end, ...]
        H_pred = H[:,self.seq_len:,...]
        H_seq = H[:,0:self.seq_len, ...]
        
        
        index = np.random.choice(M, self.length, replace = False)    
        H = H[index, ...]
        H_seq = H_seq[index, ...]
        H_pred = H_pred[index, ...] #shape: L \times M \times Nr \times Nt
        
        
        return H, H_seq, H_pred


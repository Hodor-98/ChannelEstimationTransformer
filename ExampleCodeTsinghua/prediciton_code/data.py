#=======================================================================================================================
#=======================================================================================================================
import os
import math
import time
import numpy as np
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft 
pi = np.pi

def Delay2Freq(H):
    '''
    H : T * M * Nr * Nt
    '''
    T, M, Nr, Nt = H.shape
    H_freq = ifft(H, axis=1) * np.sqrt(M)
    return H_freq

def Freq2Delay(H):
    '''
    H : T * M * Nr * Nt
    '''
    T, M, Nr, Nt = H.shape
    H_delay = fft(H, axis=1) / np.sqrt(M)
    return H_delay

def Freq2Delay_Ext_Supp(H_freq):
    '''
    H_freq: T*M*Nr*Nt
    '''
    T, M, Nr, Nt = H_freq.shape
    H_delay = Freq2Delay(H_freq)
    
    #时延域排序
    temp = np.mean(np.abs(H_delay), (0,2,3))
    index = np.argsort(temp)[::-1]

    #抽取最大的能量分量
    th = 0.01 * temp[ index[1] ]
    N = np.sum( temp > th )
    N = 100

    # N = N_max if N > N_max else N 
    index0 = index[N:]

    H_delay2 = H_delay.copy()
    H_delay2[:, index0, :, :] = 0
    H2 = Delay2Freq(H_delay2)

    plt.figure 
    plt.plot(H2[:,0,0,0])
    plt.plot(H_freq[:,0,0,0],'--')
    plt.savefig('test.png')

    plt.figure(2)
    plt.plot(H_delay[0,:,0,0])
    plt.plot(H_delay[15,:,0,0])
    plt.savefig('test2.png')
    print('Lovelive')


def Freq2Delay_Given_Supp(H, support):
    pass 

def LoadBatch(H, mode = 'test'):
    '''
    H: T * M * Nr * Nt
    ''' 
    if mode == 'test':
        T, M, Nr, Nt = H.shape 
        H = np.transpose(H, [1,0,2,3])
        H = H.reshape([M, T, Nr * Nt])
        H_real = np.zeros([M, T, Nr * Nt, 2])
        H_real[:,:,:,0] = H.real 
        H_real[:,:,:,1] = H.imag 
        H_real = H_real.reshape([M, T, Nr*Nt*2])
        H_real = torch.tensor(H_real, dtype = torch.float32)
    else:
        B, T, M, Nr, Nt = H.shape 
        # H = H.transpose(0,2,1,3,4)
        H = H.transpose(1,2)
        H = H.reshape([B*M, T, Nr * Nt])
        H_real = torch.zeros([B*M, T, Nr * Nt, 2])
        H_real[:,:,:,0] = H.real 
        H_real[:,:,:,1] = H.imag 
        H_real = H_real.reshape([B*M, T, Nr*Nt*2])
        H_real = torch.tensor(H_real, dtype = torch.float32)
    return H_real






def getdata(path, scale = 1):
    channel = scio.loadmat(path)['channel']
    data = channel['data'].item()
    sr = channel['SampleRate'].item()
    return data, sr

def noise(H, SNR):
    
    sigma = 10 ** (- SNR / 10) 
    noise = np.sqrt( sigma / 2 ) *  (np.random.randn( *H.shape ) + 1j * np.random.randn(*H.shape))
    # 归一化
    noise = noise * np.sqrt(np.mean(np.abs(H)**2))
    return H + noise



def channelnorm(H):
    H = H / np.sqrt(np.mean(np.abs(H)**2))
    return H
    
#=======================================================================================================================
#=======================================================================================================================
class SeqData(Dataset):
    def __init__(self, path, prev_len,  pred_len, mode='test', SNR = 14, ir = 1, samples = 1, v_min = 30, v_max = 60):
        self.prev_len = prev_len
        self.mode = mode
        self.pred_len = pred_len 
        self.samples = self.pred_len + self.prev_len
        
        self.ir = ir
        self.length = (self.samples - 1) * self.ir + 1 #总长度
        self.SNR = SNR
        self.path = path
        self.datapath_list = os.listdir(self.path)
        self.datapath_list = [i for i in self.datapath_list if ('.mat' in i and 'CDL' in i)] 
        self.datapath = []
        for i in range(len(self.datapath_list)):
            temp = self.datapath_list[i]
            print(temp)
            speed = int((temp.split('v')[1]).split('_')[0])
            if v_max >= speed >= v_min:
                self.datapath.append(temp)
                # print(speed)
        self.samples = samples


    def __getitem__(self, index):
        seed = math.floor(math.modf(time.time())[0]*500*320000)**2 % (2**32 - 2)
        np.random.seed(seed)
        filepath = os.path.join(self.path, self.datapath[index])
        H, sr = getdata( filepath )
        T,M,Nr,Nt = H.shape  #slot数 * 子载波数 * 基站天线数 * 用户天线数
        L = self.length       #序列长度
        
        start = np.random.randint(0, T-L+1) #序列开始位置
        end = start + L

        H = H[start:end, ...]
        
        #归一化
        H = channelnorm(H)

        #加噪声
        H = noise(H, self.SNR)

        #切分
        H_sample = H[0::self.ir, ...]
        H_pred = H_sample[self.prev_len:,...]
        H_prev = H_sample[0:self.prev_len, ...]

        if self.mode == 'train':
        # if self.mode:
            index = np.random.choice(M, self.samples, replace = False)    
            H = H[:, index, :, :]
            H_sample = H_sample[:, index, :, :]
            H_prev = H_prev[:, index, :, :]
            H_pred = H_pred[:, index, :, :] #shape: L \times M \times Nr \times Nt

        return H, H_sample, H_prev, H_pred
        # return H, H_sample
    
    def __len__(self):
        return len(self.datapath)




if __name__ == '__main__':
    path = '/data/CuiMingyao/Project/5GmmWave_informer_v3/data_CDL_B_30/test'

    test_dataset = SeqData(path, prev_len=20,  pred_len=3, SNR = 50)



    data, data_prev, data_pred = test_dataset[0]
    # data_prev = LoadBatch(data_prev)

    H_delay = Freq2Delay_Ext_Supp(data)

    L, M, Nr, Nt = data.shape
    
    x = np.arange(L).reshape([L,1])
    plt.figure(1)
    plt.grid(True)
    plt.box(True)    
    plt.plot(x, data[:, 0,0,0], 'r--')
    plt.savefig('test.png')


    print('Lovelive')
#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
#=======================================================================================================================
#=======================================================================================================================




# NMSE Function Defining
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse
def Score(NMSE):
    score = 1 - NMSE
    return score


class Adap_MSELoss(nn.Module):
    def __init__(self, M=1, N=8):
        super(Adap_MSELoss, self).__init__()
        self.M = M
        self.N = N

    def forward(self, x_hat, x):
        nmse = self.Adap_MSE(x_hat, x)
        return nmse

    def Adap_MSE(self,x_hat, x):
        shape = list(x_hat.shape)[0:-1]
        shape.extend([2,self.M, self.N])
        x_hat = x_hat.reshape(shape)
        x = x.reshape(shape)
    
        res = (x - x_hat)**2
        softres = torch.softmax(res, dim = -1)
        nmse = torch.sum(softres * res * self.N)
        return nmse

class Adap_NMSELoss(nn.Module):
    def __init__(self, M=1, N=8):
        super(Adap_NMSELoss, self).__init__()
        self.M = M
        self.N = N

    def forward(self, x_hat, x):
        nmse = self.Adap_NMSE(x_hat, x)
        return nmse

    def Adap_NMSE(self,x_hat, x):
        shape = list(x_hat.shape)[0:-1]
        shape.extend([2,self.M, self.N])
        x_hat = x_hat.reshape(shape)
        x = x.reshape(shape)
    
        res = (x - x_hat)**2
        power = x**2
        nres = res/power
        softnres = torch.softmax(res, dim = -1)
        nmse = torch.sum(softnres * res * self.N)/torch.sum(power)
        return nmse

def NMSE_cuda(x_hat, x):
    # x_real = x[:, :, :, 0].view(len(x),-1) - 0.5
    # x_imag = x[:, :, :, 1].view(len(x),-1) - 0.5
    # x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1) - 0.5
    # x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1) - 0.5
    # power = torch.sum(x_real**2 + x_imag**2, 1)
    # mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, 1)
    # nmse = mse/power
    power = torch.sum(x**2)
    mse = torch.sum((x-x_hat)**2)
    nmse = mse/power
    return nmse
    
class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse
import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
import argparse
from torch.utils.data import Dataset, DataLoader
import math
from torch import nn, Tensor
from models.model import Informer, InformerStack, LSTM,RNN,GRU, InformerStack_e2e
import matplotlib.pyplot as plt
from metrics import NMSELoss, Adap_NMSELoss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

enc_in = 16
dec_in = 16
c_out = 16
seq_len = 25
label_len = 10
pred_len = 5
factor = 5
d_model = 64
n_heads = 8
e_layers = 4
d_layers = 3
d_ff = 64
dropout = 0.05
attn = 'full'
embed = 'fixed'
activation = 'gelu'
output_attention = True
distil = True
device = torch.device('cuda')# Example value, replace this with your device choice


model = InformerStack(
    enc_in,
    dec_in,
    c_out,
    seq_len,
    label_len,
    pred_len,
    factor,
    d_model,
    n_heads,
    e_layers,
    d_layers,
    d_ff,
    dropout,
    attn,
    embed,
    activation,
    output_attention,
    distil,
    device
)

model.eval()
state_dict = torch.load("best_model_params.pt")
state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
print("informer has been loaded!")
informer = torch.nn.DataParallel( model ).cuda()



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

criterion = NMSELoss()
def evaluate(model):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        with open('GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V30__validate.pickle', 'rb') as handle:
            channel = pickle.load(handle)
        
        data, label = LoadBatch(channel[:, :25, :, :]), LoadBatch(channel[:, -5:, :, :])
        
        inp_net = data.to(device)
        label = label.to(device)

        enc_inp = inp_net
        dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
        dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)
        seq_len_temp = data.size(0)
        output = model(enc_inp,dec_inp)[0]
        total_loss += seq_len_temp * criterion(output, label).item()
    return total_loss / (channel.size(0) - 1)


with open('GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V30__validate.pickle', 'rb') as handle:
    channel = pickle.load(handle)
    
print(channel.shape)
batch_size, M, Nr, Nt = channel.shape 
    
data = LoadBatch(channel[:, :25, :, :])
inp_net = data.to(device)

enc_inp = inp_net
dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)

# informer
if output_attention:
    outputs_informer = informer(enc_inp, dec_inp)[0]
else:
    outputs_informer = informer(enc_inp, dec_inp)
outputs_informer = outputs_informer.cpu().detach()
outputs_informer = real2complex(np.array(outputs_informer))


outputs_informer = outputs_informer.reshape([batch_size, pred_len, Nr, Nt])

x = np.array(list(range(channel.shape[1])))

for j in range(channel.shape[0]):
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x[-pred_len:],outputs_informer[j,:,i,0].real)
        plt.plot(x,channel[j,:,i,0].real, linestyle='--')
    plt.savefig(f"ChannelPredictionPlots/Prediction_{j}.png", dpi=300)
    
test_loss = evaluate(informer)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
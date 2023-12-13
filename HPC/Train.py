#!/usr/bin/env python3
#SBATCH --partition=elec.gpu.q
#SBATCH --output=openme.out
#SBATCH --gres=gpu:1

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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


with open('GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V30__0.pickle', 'rb') as handle:
    train_data = pickle.load(handle)



bptt = 35
def get_batch(source, i):

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target



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
device = 'cpu'  # Example value, replace this with your device choice


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


criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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


def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 20
    start_time = time.time()

    num_batches = 29
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        
        with open(f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V30__{batch}.pickle', 'rb') as handle:
            channel = pickle.load(handle)
        print(channel.shape)
        
        data, label = LoadBatch(channel[:, :25, :, :]), LoadBatch(channel[:, -5:, :, :])
        
        print(label.shape)
        inp_net = data.to(device)

        enc_inp = inp_net
        dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
        dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)
        

        
        output = model(enc_inp,dec_inp)[0]
        print(output.shape)
        print(label.shape)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            
def evaluate(model):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        with open('GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V30__29.pickle', 'rb') as handle:
            channel = pickle.load(handle)
        
        data, label = LoadBatch(channel[:, :25, :, :]), LoadBatch(channel[:, -5:, :, :])
        
        inp_net = data.to(device)

        enc_inp = inp_net
        dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
        dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)
        seq_len_temp = data.size(0)
        output = model(enc_inp,dec_inp)[0]
        total_loss += seq_len_temp * criterion(output, label).item()
    return total_loss / (channel.size(0) - 1)

best_val_loss = float('inf')
epochs = 10
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), best_model_params_path)

    scheduler.step()
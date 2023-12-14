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
model = torch.nn.DataParallel(model).cuda()


model.load_state_dict(torch.load("best_model_params.pt"))

# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = NMSELoss()

# MSELoss = nn.MSELoss()
lr = 1  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

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

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 256
    start_time = time.time()

    with open(f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V30.pickle', 'rb') as handle:
        dataset = pickle.load(handle)
    print(type(dataset))

    print(f"Info: Dataset contains {dataset.size(0)} batches")
    num_batches = dataset.size(0)
    for batch, i in enumerate(range(0, dataset.size(0) - 1)):
        channel = dataset[batch]

        
        data, label = LoadBatch(channel[:, :25, :, :]), LoadBatch(channel[:, -5:, :, :])
        label = label.to(device)

        enc_inp = data.to(device)
        dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
        dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)
    
        output = model(enc_inp,dec_inp)[0]
        loss = criterion(output, label)
        # print(MSELoss(output,label).item())
        # loss = MSELoss(output,label)
        
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
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}', flush=True)
            total_loss = 0
            start_time = time.time()

            
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
 

best_val_loss = float('inf')
epochs = 100
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89, flush=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_params.pt")

    scheduler.step()
    
model.load_state_dict(torch.load("best_model_params.pt"))
test_loss = evaluate(model)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)


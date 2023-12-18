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
from data import LoadBatch, real2complex 
import matplotlib.pyplot as plt
from metrics import NMSELoss, Adap_NMSELoss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

speed = 30
lr = 1  # learning rate
epochs = 10
direction = 'uplink'
max_batches = 2**30

# Check GPU memory i.e how much memory is there, how much is free
def check_gpu_memory():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    else:
        print("No GPU available.")



    
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True  # Enable cuDNN
    torch.backends.cudnn.benchmark = True  # Use cuDNN's auto-tuner for the best performance
    torch.cuda.init()
    check_gpu_memory()
    force_cudnn_initialization()

    print(torch.cuda.device_count())

enc_in = 16
hs = 256
hl = 2
seq_len = 25
label_len = 10
pred_len = 5
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
device_ids = [i for i in range(torch.cuda.device_count())]


# lstm = LSTM(enc_in, enc_in, hs, hl) 
rnn =  RNN(enc_in, enc_in, hs, hl) 
# gru = GRU(enc_in, enc_in, hs, hl) 







def train(model: nn.Module,optimizer,scheduler,epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    criterion = NMSELoss()

    start_time = time.time()

    with open(f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V{speed}_{direction}.pickle', 'rb') as handle:
        dataset = pickle.load(handle)
    print(dataset.shape)
    print(f"Info: Dataset contains {dataset.size(0)} batches")
    num_batches = dataset.size(0)
    if max_batches < num_batches:
        num_batches = max_batches
        
    log_interval = num_batches/8
    for batch, i in enumerate(range(0, num_batches - 1)):
        channel = dataset[batch]
        

        data, label = LoadBatch(channel[:, :30, :, :]), LoadBatch(channel[:, -5:, :, :])
        label = label.to(device)

        enc_inp = data.to(device)


    
        output = model.train_data(enc_inp, device)
        print(output.shape)
        loss = criterion(output[:, -5:, :], label)
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
    criterion = NMSELoss()
    
    with torch.no_grad():
        with open(f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V{speed}_{direction}__validate.pickle', 'rb') as handle:
            channel = pickle.load(handle)
        channel = channel[0]
        print(channel.shape)
        
        data, label = LoadBatch(channel[:, :30, :, :]), LoadBatch(channel[:, -5:, :, :])
        label = label.to(device)

        enc_inp = data.to(device)


        seq_len_temp = data.size(0)
        output = model.train_data(enc_inp, device)
        total_loss += seq_len_temp * criterion(output[:, -5:, :], label).item()
        
    return total_loss / (channel.size(0) - 1)


def train_loop(model: nn.Module):
    best_val_loss = float('inf')
    ModelDictName = f"TrainedTransformers/{model.__class__.__name__}_best_model_params_V{speed}_{direction}.pt"

    if os.path.exists(ModelDictName):
        model.load_state_dict(torch.load(ModelDictName,map_location="cpu"))
        print("Model loaded successfully!")
    else:
        print(f"File '{ModelDictName}' does not exist. Creating new model.")
        

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel( model ).cuda() if torch.cuda.is_available() else model 

    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model,optimizer,scheduler,epoch)
        val_loss = evaluate(model)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89, flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ModelDictName)

        scheduler.step()



train_loop(rnn)
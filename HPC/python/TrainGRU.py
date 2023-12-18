import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import numpy as np
import pickle
import math
from models.model import Informer, InformerStack, LSTM, RNN, GRU, InformerStack_e2e
from metrics import NMSELoss, Adap_NMSELoss
from data import SeqData
from torch.utils.data import Dataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constants
speed = 30
lr = 1  # learning rate
epochs = 1
direction = 'uplink'
max_batches = 2 ** 30
enc_in = 16
hs = 256
hl = 2
seq_len = 25
label_len = 10
pred_len = 5
use_gpu = False
device = torch.device('cuda:0') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
device_ids = list(range(torch.cuda.device_count())) if (torch.cuda.is_available() and use_gpu) else []

# Check GPU memory
def check_gpu_memory():
    if (torch.cuda.is_available() and use_gpu):
        current_device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    else:
        print("No GPU available.")

# Force cuDNN initialization
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

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

# Function to train the model using SeqData DataLoader
def train(model: nn.Module, optimizer, scheduler, epoch, dataloader) -> None:
    model.train()
    total_loss = 0.
    criterion = NMSELoss()
    start_time = time.time()
    
    batch = next(iter(dataloader))
    for itr in range(dataloader.batch_size):
        H, H_seq, H_pred = [tensor[itr] for tensor in batch]
        
        data, label = LoadBatch(H), LoadBatch(H_pred)
        data = data.to(device)
        label = label.to(device)
        
        output = model.train_data(data, device)
        loss = criterion(output[:, -pred_len:, :], label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if itr   % (dataloader.batch_size // 8) == 0 and itr > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / (dataloader.batch_size // 8)
            cur_loss = total_loss / (dataloader.batch_size // 8)
            ppl = math.exp(cur_loss)

            print(f'| epoch {epoch:3d} | {itr:5d}/{dataloader.batch_size:5d} batches | '
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
        
        data, label = LoadBatch(channel[:, :30, :, :]), LoadBatch(channel[:, -pred_len:, :, :])
        label = label.to(device)

        enc_inp = data.to(device)


        seq_len_temp = data.size(0)
        output = model.train_data(enc_inp, device)
        total_loss += seq_len_temp * criterion(output[:, -pred_len:, :], label).item()
        
    return total_loss / (channel.size(0) - 1)


# Function to run the training loop using SeqData DataLoader
def train_loop(model: nn.Module, dataloader):
    best_val_loss = float('inf')
    model_dict_name = f"TrainedTransformers/{model.__class__.__name__}_best_model_params_V{speed}_{direction}.pt"

    if (torch.cuda.is_available() and use_gpu):
        model.load_state_dict(torch.load(model_dict_name, map_location="cuda"))
    else:
        if os.path.exists(model_dict_name):
            model.load_state_dict(torch.load(model_dict_name, map_location="cpu"))
            print("Model loaded successfully!")
        else:
            print(f"File '{model_dict_name}' does not exist. Creating a new model.")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

    model = torch.nn.DataParallel(model).cuda() if (torch.cuda.is_available() and use_gpu) else model

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, optimizer, scheduler, epoch, dataloader)
        val_loss = evaluate(model)  # You should modify evaluate() to use SeqData DataLoader for validation as well
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89, flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dict_name)

        scheduler.step()

# Execution
if (torch.cuda.is_available() and use_gpu):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.init()
    check_gpu_memory()
    force_cudnn_initialization()
    print(torch.cuda.device_count())

# Initialize and run training loop with SeqData DataLoader
dataset_name = f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V{speed}_{direction}.pickle'
testData =  SeqData(dataset_name, seq_len, pred_len)
dataloader = DataLoader(dataset = testData, batch_size = 512, shuffle = True,  
                          num_workers = 4, drop_last = False, pin_memory = True) 

gru = GRU(enc_in, enc_in, hs, hl)
train_loop(gru, dataloader)

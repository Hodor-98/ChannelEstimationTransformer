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
from data import SeqData
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SNR = 15
speed = 30
lr = 1  # learning rate
epochs = 20
direction = 'uplink'
num_batches = 10
channel_model = "CDLBFixedDirection"


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
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
use_gpu = True

# Define parameter names and values
parameters = [
    ("Encoder Input Size (enc_in)", enc_in),
    ("Decoder Input Size (dec_in)", dec_in),
    ("Output Size (c_out)", c_out),
    ("Sequence Length (seq_len)", seq_len),
    ("Label Length (label_len)", label_len),
    ("Prediction Length (pred_len)", pred_len),
    ("Scaling Factor (factor)", factor),
    ("Model Dimension (d_model)", d_model),
    ("Number of Attention Heads (n_heads)", n_heads),
    ("Encoder Layers (e_layers)", e_layers),
    ("Decoder Layers (d_layers)", d_layers),
    ("Feed-Forward Dimension (d_ff)", d_ff),
    ("Dropout Probability (dropout)", dropout),
    ("Attention Mechanism (attn)", attn),
    ("Embedding Type (embed)", embed),
    ("Activation Function (activation)", activation),
    ("Output Attention? (output_attention)", output_attention),
    ("Distillation Used? (distil)", distil),
    ("Channel Model", channel_model),
]

# Calculate the maximum width for alignment
max_width = max(len(name) for name, _ in parameters) + 5

# Print the table
print("Transformer Model Configuration:")
for name, value in parameters:
    print(f"{name.ljust(max_width)}{value}")


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

device_ids = [i for i in range(torch.cuda.device_count())]



model_dict_name = f"TrainedTransformers/{model.__class__.__name__}_{channel_model}_{factor}x_d_model{d_model}_n_heads{n_heads}_e_layers{e_layers}_d_layers{d_layers}_d_ff{d_ff}_dropout{dropout}_attn_{attn}_embed_{embed}_activation_{activation}_enc_in{enc_in}_dec_in{dec_in}_c_out{c_out}_seq_len{seq_len}_label_len{label_len}_pred_len{pred_len}_output_attention{output_attention}_distil{distil}_{direction}.pt"
 
if os.path.exists(model_dict_name):
    if (torch.cuda.is_available() and use_gpu):
        state_dict = torch.load(model_dict_name,map_location=torch.device('cuda'))
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    else:
        state_dict = torch.load(model_dict_name,map_location=torch.device('cpu'))
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        print("Model loaded successfully!")
else:
    print(f"Model does not yet exist! Creating new model")

model = torch.nn.DataParallel( model ).cuda() if torch.cuda.is_available() else model 

criterion = NMSELoss()
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

dataset_name = f'GeneratedChannels/Channel{channel_model}_Tx4_Rx2_DS1e-07_V{speed}_{direction}.pickle'
testData =  SeqData(dataset_name, seq_len, pred_len, SNR=SNR)
test_loader = DataLoader(dataset = testData, batch_size = 512*16, shuffle = True,  
                          num_workers = 4, drop_last = False, pin_memory = True) 

evaluateDatasetName = f'GeneratedChannels/Channel{channel_model}_Tx4_Rx2_DS1e-07_V{speed}_{direction}__validate.pickle'
evaluateData =  SeqData(evaluateDatasetName, seq_len, pred_len, SNR=SNR)
EvaluaterLoader = DataLoader(evaluateData, batch_size=8, shuffle=True)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.

    start_time = time.time()


        
    log_interval = test_loader.batch_size/8
    for batch, i in enumerate(range(test_loader.batch_size-1)):

        
        H, H_seq, H_pred = testData[batch]
        

        data, label = LoadBatch(H_seq), LoadBatch(H_pred)
        label = label.to(device)

        enc_inp = data.to(device)
        
        
        dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
        dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)
    
        output = model(enc_inp,dec_inp)[0]
        loss = criterion(label,output)
        # print(MSELoss(output,label).item())
        # loss = MSELoss(output,label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)

            print(f'| epoch {epoch:3d} | {batch:5d}/{test_loader.batch_size:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}', flush=True)
            total_loss = 0
            start_time = time.time()

            
def evaluate(model):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        batch = next(iter(EvaluaterLoader))
        
        total_loss = 0
        for itr in range(EvaluaterLoader.batch_size):
            H, H_seq, H_pred = [tensor[itr] for tensor in batch]
            
            data, label = LoadBatch(H_seq), LoadBatch(H_pred)
            
            inp_net = data.to(device)
            label = label.to(device)

            enc_inp = inp_net
            dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
            dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)
            output = model(enc_inp,dec_inp)[0]

            total_loss += criterion(label,output).item()
            
            outputs_plot_validate = real2complex(np.array(output.detach().cpu()))
            x = np.array(list(range(seq_len+pred_len)))
            
            plt.figure()
            for i in range(4):
                plt.subplot(2,2,i+1)
                plt.plot(x[-pred_len:],outputs_plot_validate[0,:,i*2].real)
                # plt.plot(x,data[0,:,i].real, linestyle='--')
                plt.plot(x,H[0,:seq_len+pred_len,i,0].real, linestyle='--')
            plt.savefig(f"ChannelPredictionPlots/{channel_model}_Prediction_{model.__class__.__name__}_{itr}.png", dpi=300)
            plt.close()
        
    return total_loss/itr
 

best_val_loss = float('inf')

torch.backends.cudnn.benchmark = True
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
        torch.save(model.state_dict(), model_dict_name)

    scheduler.step()
    
model.load_state_dict(torch.load(model_dict_name))
test_loss = evaluate(model)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)


import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
import argparse
import math
from torch.utils.data import Dataset, DataLoader
from models.model import Informer, InformerStack, LSTM, RNN, GRU, InformerStack_e2e
from metrics import NMSELoss, Adap_NMSELoss
from utils import *
from data import SeqData

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



speed = 30
direction = 'uplink'
channel_model = "CDLBFixedDirection"

# Constants
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

hs = 256
hl = 2

use_gpu = True


check_gpu_memory(use_gpu= use_gpu)
force_cudnn_initialization()

device = torch.device('cuda:0') if use_gpu else torch.device('cpu')   # Example value, replace this with your device choice

transformer = InformerStack(
    enc_in, dec_in, c_out, seq_len, label_len, pred_len, factor, d_model, n_heads,
    e_layers, d_layers, d_ff, dropout, attn, embed, activation, output_attention,
    distil, device
)

lstm = LSTM(enc_in, enc_in, hs, hl) 
rnn =  RNN(enc_in, enc_in, hs, hl) 
gru = GRU(enc_in, enc_in, hs, hl) 


transformer_dict_name = f"TrainedTransformers/{transformer.__class__.__name__}_{channel_model}_{factor}x_d_model{d_model}_n_heads{n_heads}_e_layers{e_layers}_d_layers{d_layers}_d_ff{d_ff}_dropout{dropout}_attn_{attn}_embed_{embed}_activation_{activation}_enc_in{enc_in}_dec_in{dec_in}_c_out{c_out}_seq_len{seq_len}_label_len{label_len}_pred_len{pred_len}_output_attention{output_attention}_distil{distil}_{direction}.pt"
lstm_dict_name = f"TrainedTransformers/{lstm.__class__.__name__}_enc{enc_in}_hs{hs}_hl{hl}_seq{seq_len}_label{label_len}_best_model_params_V{speed}_{direction}.pt"
rnn_dict_name = f"TrainedTransformers/{rnn.__class__.__name__}_enc{enc_in}_hs{hs}_hl{hl}_seq{seq_len}_label{label_len}_best_model_params_V{speed}_{direction}.pt"
gru_dict_name = f"TrainedTransformers/{gru.__class__.__name__}_enc{enc_in}_hs{hs}_hl{hl}_seq{seq_len}_label{label_len}_best_model_params_V{speed}_{direction}.pt"


state_dict = torch.load(transformer_dict_name, map_location=torch.device('cpu'))
state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
transformer.load_state_dict(state_dict)
transformer = torch.nn.DataParallel( transformer ).cuda() if use_gpu else transformer 
print("transformer has been loaded!")

# load LSTM 
state_dict = torch.load(lstm_dict_name)
lstm.load_state_dict(state_dict) 
print("lstm  has been loaded!") 
lstm = lstm.cuda() if use_gpu else lstm 
 
# load gru 
state_dict = torch.load(gru_dict_name)
gru.load_state_dict(state_dict) 
print("lstm  has been loaded!") 
gru = gru.cuda() if use_gpu else gru 
 
# load rnn 
state_dict = torch.load(rnn_dict_name)
rnn.load_state_dict(state_dict) 
print("rnn has been loaded!") 
rnn = rnn.cuda() if use_gpu else rnn 
 
transformer.eval() 
lstm.eval() 
gru.eval() 
rnn.eval() 

NMSE0 = np.zeros(pred_len + 1) 
NMSE1 = np.zeros(pred_len + 1) 
NMSE2 = np.zeros(pred_len + 1) 
NMSE3 = np.zeros(pred_len + 1) 
NMSE4 = np.zeros(pred_len + 1) 
NMSE5 = np.zeros(pred_len + 1) 
NMSE6 = np.zeros(pred_len + 1) 
NMSE7 = np.zeros(pred_len + 1) 
NMSE8 = np.zeros(pred_len + 1) 



criterion = NMSELoss() 

evaluateDatasetName = f'GeneratedChannels/Channel{channel_model}_Tx4_Rx2_DS1e-07_V{speed}_{direction}__validate.pickle'
evaluateData =  SeqData(evaluateDatasetName, seq_len, pred_len)
EvaluaterLoader = DataLoader(evaluateData, batch_size=8, shuffle=False)


    
num_itr = EvaluaterLoader.batch_size

channel_dataset = next(iter(EvaluaterLoader))

for itr in range(num_itr):
    H, H_seq, H_pred = [tensor[itr] for tensor in channel_dataset]
    print(H.shape)
    
    batch_size, M, Nr, Nt = H.shape 
        
    data, label = LoadBatch(H_seq), LoadBatch(H_pred)


    inp_net = data.to(device)

    enc_inp = inp_net
    dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
    dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)

    outputs_informer = transformer(enc_inp, dec_inp)[0]
    outputs_informer = outputs_informer.cpu().detach()
    nmse_informer = criterion(outputs_informer, label) 
    outputs_informer = real2complex(np.array(outputs_informer))
    
    print(outputs_informer.shape)
    
    
    outputs_lstm = lstm.test_data(enc_inp, pred_len, device) 
    outputs_lstm = outputs_lstm.cpu().detach() 
    nmse_lstm = criterion(outputs_lstm, label) 
    outputs_lstm = real2complex(np.array(outputs_lstm))   # shape = [64, 3, 8] 

    outputs_rnn = rnn.test_data(enc_inp, pred_len, device) 
    outputs_rnn = outputs_rnn.cpu().detach() 
    nmse_rnn = criterion(outputs_rnn, label) 
    outputs_rnn = real2complex(np.array(outputs_rnn))   # shape = [64, 3, 8
    
    outputs_gru = gru.test_data(enc_inp, pred_len, device) 
    outputs_gru = outputs_gru.cpu().detach() 
    nmse_gru = criterion(outputs_gru, label) 
    outputs_gru = real2complex(np.array(outputs_gru))   # shape = [64, 3, 8
    
    
    ''' 
    reshape 
    ''' 
    outputs_informer = outputs_informer.reshape([batch_size, pred_len, Nr, Nt])  # shape = [64, 3, 4, 2] 
    outputs_lstm = outputs_lstm.reshape([batch_size, pred_len, Nr, Nt]) 
    outputs_gru = outputs_gru.reshape([batch_size, pred_len, Nr, Nt]) 
    outputs_rnn = outputs_rnn.reshape([batch_size, pred_len, Nr, Nt]) 
 

    for s in range(pred_len+ 1): 
        H_true = H_pred[:, s-1, :, :] if s > 0 else  data[:, -pred_len - 1, ...] 
        H_hat = outputs_informer[:, s-1, :, :] if s > 0 else data[:, -pred_len - 1, ...] 
        error = torch.sum(np.abs(H_true - H_hat)**2) 
        power = torch.sum(np.abs(H_true)**2) 
        NMSE0[s] += error/power/num_itr 

    # lstm 
    for s in range(pred_len+ 1): 
        H_true = H_pred[:, s-1, :, :] if s > 0 else  data[:, -pred_len - 1, ...] 
        H_hat = outputs_lstm[:, s-1, :, :] if s > 0 else data[:, -pred_len - 1, ...] 
        error = torch.sum(np.abs(H_true - H_hat)**2) 
        power = torch.sum(np.abs(H_true)**2) 
        NMSE1[s] += error/power/num_itr 

    # gru 
    for s in range(pred_len+ 1): 
        H_true = H_pred[:, s-1, :, :] if s > 0 else  data[:, -pred_len - 1, ...] 
        H_hat = outputs_gru[:, s-1, :, :] if s > 0 else data[:, -pred_len - 1, ...] 
        error = torch.sum(np.abs(H_true - H_hat)**2) 
        power = torch.sum(np.abs(H_true)**2) 
        NMSE2[s] += error/power/num_itr 

    # rnn 
    for s in range(pred_len+ 1): 
        H_true = H_pred[:, s-1, :, :] if s > 0 else  data[:, -pred_len - 1, ...] 
        H_hat = outputs_rnn[:, s-1, :, :] if s > 0 else data[:, -pred_len - 1, ...]    
        error = torch.sum(np.abs(H_true - H_hat)**2) 
        power = torch.sum(np.abs(H_true)**2) 
        NMSE3[s] += error/power/num_itr 



plt.figure() 
plt.plot(10*np.log10(NMSE0)) 
plt.plot(10*np.log10(NMSE1)) 
plt.plot(10*np.log10(NMSE2)) 
plt.plot(10*np.log10(NMSE3)) 

plt.legend(['Transformer', 'LSTM', 'GRU', 'RNN']) 
plt.xlabel('SRS (0.625 ms)') 
plt.ylabel('NMSE (dB)') 
plt.savefig('NMSE.png', dpi = 300) 

print("DONE")
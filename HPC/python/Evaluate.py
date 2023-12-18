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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

speed = 30
direction = 'uplink'

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

use_gpu = False


device = torch.device('cuda:0') if use_gpu else torch.device('cpu')   # Example value, replace this with your device choice

transformer = InformerStack(
    enc_in, dec_in, c_out, seq_len, label_len, pred_len, factor, d_model, n_heads,
    e_layers, d_layers, d_ff, dropout, attn, embed, activation, output_attention,
    distil, device
)

lstm = LSTM(enc_in, enc_in, hs, hl) 
rnn =  RNN(enc_in, enc_in, hs, hl) 
gru = GRU(enc_in, enc_in, hs, hl) 



state_dict = torch.load(f"TrainedTransformers/{transformer.__class__.__name__}_best_model_params_V{speed}_{direction}.pt", map_location=torch.device('cpu'))
state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
transformer.load_state_dict(state_dict)
transformer = torch.nn.DataParallel( lstm ).cuda() if use_gpu else transformer 
print("transformer has been loaded!")

# load LSTM 
state_dict = torch.load(f"TrainedTransformers/{lstm.__class__.__name__}_best_model_params_V{speed}_{direction}.pt")
lstm.load_state_dict(state_dict) 
print("lstm  has been loaded!") 
lstm = torch.nn.DataParallel( lstm ).cuda() if use_gpu else lstm 
 
# load gru 
state_dict = torch.load(f"TrainedTransformers/{gru.__class__.__name__}_best_model_params_V{speed}_{direction}.pt")
gru.load_state_dict(state_dict) 
print("lstm  has been loaded!") 
gru = torch.nn.DataParallel( gru ).cuda() if use_gpu else gru 
 
# load rnn 
state_dict = torch.load(f"TrainedTransformers/{rnn.__class__.__name__}_best_model_params_V{speed}_{direction}.pt")
rnn.load_state_dict(state_dict) 
print("rnn has been loaded!") 
rnn = torch.nn.DataParallel( rnn ).cuda() if use_gpu else rnn 
 
transformer.eval() 
lstm.eval() 
gru.eval() 
rnn.eval() 


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
        with open(f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V{speed}_{direction}__validate.pickle', 'rb') as handle:
            channel = pickle.load(handle)
        channel = channel[0]
        
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


with open(f'GeneratedChannels/ChannelCDLB_Tx4_Rx2_DS1e-07_V{speed}_{direction}__validate.pickle', 'rb') as handle:
    channel = pickle.load(handle)
    channel = channel[0]
    

batch_size, M, Nr, Nt = channel.shape 
    
data, label = LoadBatch(channel[:, :25, :, :]), LoadBatch(channel[:, -5:, :, :])


inp_net = data.to(device)

enc_inp = inp_net
dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(device)
dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)

# informer
if output_attention:
    outputs_informer = transformer(enc_inp, dec_inp)[0]
    output = outputs_informer
else:
    outputs_informer = transformer(enc_inp, dec_inp)
outputs_informer = outputs_informer.cpu().detach()
outputs_informer = real2complex(np.array(outputs_informer))


outputs_informer = outputs_informer.reshape([batch_size, pred_len, Nr, Nt])

x = np.array(list(range(channel.shape[1])))

for j in range(8):
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x[-pred_len:],outputs_informer[j,:,i,0].real)
        plt.plot(x,channel[j,:,i,0].real, linestyle='--')
    plt.savefig(f"ChannelPredictionPlots/Prediction_transformer_{j}.png", dpi=300)
    
    
    
test_loss = evaluate(transformer)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)


# data, label = LoadBatch(channel[:, :25, :, :]), LoadBatch(channel[:, -5:, :, :])

# lstm 
outputs_lstm = lstm.test_data(enc_inp, pred_len, device) 
outputs_lstm = outputs_lstm.cpu().detach() 
nmse_lstm = criterion(outputs_lstm, label) 
outputs_lstm = real2complex(np.array(outputs_lstm))   # shape = [64, 3, 8] 

print(outputs_lstm.shape)

for j in range(8):
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x[-pred_len:],outputs_lstm[j,:,i].real)
        plt.plot(x,channel[j,:,i,0].real, linestyle='--')
    plt.savefig(f"ChannelPredictionPlots/Prediction_lstm_{j}.png", dpi=300)
    
# gru 
outputs_gru = gru.test_data(enc_inp, pred_len, device) 
outputs_gru = outputs_gru.cpu().detach() 
nmse_gru = criterion(outputs_gru, label) 
outputs_gru = real2complex(np.array(outputs_gru))


for j in range(8):
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x[-pred_len:],outputs_gru[j,:,i].real)
        plt.plot(x,channel[j,:,i,0].real, linestyle='--')
    plt.savefig(f"ChannelPredictionPlots/Prediction_gru_{j}.png", dpi=300)
    
    

# rnn 
outputs_rnn = rnn.test_data(enc_inp, pred_len, device) 
outputs_rnn = outputs_rnn.cpu().detach() 
nmse_rnn = criterion(outputs_rnn, label) 
outputs_rnn = real2complex(np.array(outputs_rnn)) 


for j in range(8):
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x[-pred_len:],outputs_rnn[j,:,i].real)
        plt.plot(x,channel[j,:,i,0].real, linestyle='--')
    plt.savefig(f"ChannelPredictionPlots/Prediction_rnn_{j}.png", dpi=300)
    

print(nmse_lstm,nmse_gru,nmse_rnn)


criterion = NMSELoss()
label = label.to(device)
NMSE = [ criterion(output[i], label[i]) for i in range(output.size(0))]

NMSE_array = np.array([tensor.item() for tensor in NMSE])

plt.figure()
plt.hist(NMSE_array,100)
plt.savefig("Histogram.png",dpi=300)
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
import matplotlib.pyplot as plt
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constants
speed = 30
lr = 1  # learning rate
SNR = 30
epochs = 10
direction = 'uplink'
max_batches = 2 ** 30
enc_in = 16
hs = 256
hl = 2
seq_len = 25
label_len = 10
pred_len = 5
use_gpu = True

channel_model = "CDLBFixedDirection"

device = torch.device('cuda:0') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
device_ids = list(range(torch.cuda.device_count())) if (torch.cuda.is_available() and use_gpu) else []

# Execution
if (torch.cuda.is_available() and use_gpu):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.init()
    check_gpu_memory(use_gpu=use_gpu)
    force_cudnn_initialization()
    print(torch.cuda.device_count())

# Initialize and run training loop with SeqData DataLoader
trainDatasetName = f'GeneratedChannels/Channel{channel_model}_Tx4_Rx2_DS1e-07_V{speed}_{direction}.pickle'
trainData =  SeqData(trainDatasetName, seq_len+15, pred_len, SNR=SNR)
trainerLoader = DataLoader(dataset = trainData, batch_size = 512*16, shuffle = True,  
                          num_workers = 4, drop_last = False, pin_memory = True)



evaluateDatasetName = f'GeneratedChannels/Channel{channel_model}_Tx4_Rx2_DS1e-07_V{speed}_{direction}__validate.pickle'
evaluateData =  SeqData(evaluateDatasetName, seq_len, pred_len, SNR=SNR)
EvaluaterLoader = DataLoader(evaluateData, batch_size=8, shuffle=True)

rnn = RNN(enc_in, enc_in, hs, hl)

model_dict_name  = f"TrainedTransformers/{rnn.__class__.__name__}_{channel_model}_enc{enc_in}_hs{hs}_hl{hl}_seq{seq_len}_label{label_len}_best_model_params_V{speed}_{direction}.pt"

if os.path.exists(model_dict_name):
    if (torch.cuda.is_available() and use_gpu):
        rnn.load_state_dict(torch.load(model_dict_name, map_location="cuda:0"))
    else:
        rnn.load_state_dict(torch.load(model_dict_name, map_location="cpu"))
    print("Model loaded successfully!")
else:
    print(f"File '{model_dict_name}' does not exist. Creating a new model.")

rnn = rnn.cuda() if use_gpu else rnn 
train_loop(rnn, trainerLoader, EvaluaterLoader, epochs, lr, model_dict_name ,device)

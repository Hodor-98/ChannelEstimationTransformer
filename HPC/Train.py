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
from models.model import Informer, InformerStack, LSTM,RNN,GRU, InformerStack_e2e
import matplotlib.pyplot as plt
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
device = 'gpu'  # Example value, replace this with your device choice


transformerModel = InformerStack(
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
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
import tensorflow as tf
import scipy.io as scio

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber


# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported.
num_ut = 1
num_bs = 1
num_ut_ant = 2
num_bs_ant = 4
# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)


rg = ResourceGrid(num_ofdm_symbols=75,
                  fft_size=16,
                  subcarrier_spacing=120e3,
                  num_tx=1,
                  num_streams_per_tx=num_streams_per_tx,        
                  cyclic_prefix_length=0,
                  num_guard_carriers=[5,6],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[0])


carrier_frequency = 28e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

ut_array = AntennaArray(num_rows=1,
                        num_cols=int(num_ut_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)


delay_spread = 100e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value. 

direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.
cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]

speed = 30/3.6            # UT speed [m/s]. BSs are always assumed to be fixed.
                      # The direction of travel will chosen randomly within the x-y plane.
                      
                      
number_of_slots = 50



# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed, max_speed = speed)
print(rg.bandwidth)
print(rg.num_time_samples)

l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max-l_min+1

batch_size = 2
num_of_batches = 10
ebno_db = 30
num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # Code rate
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
ChannelNP = []



for batch in range(num_of_batches):
    print(f"INFO: Generating batch {batch}", flush=True)
    
    for subbatch in range(int(64/batch_size)):


        # The following values for truncation are recommended.
        # Please feel free to tailor them to you needs.

        a, tau  = cdl(batch_size=batch_size, num_time_steps=rg.num_time_samples*number_of_slots+l_tot-1, sampling_frequency=rg.bandwidth)

        # This function removes nulled subcarriers from any tensor having the shape of a resource grid
        remove_nulled_scs = RemoveNulledSubcarriers(rg)

        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
            
        # We need to sub-sample the channel impulse reponse to compute perfect CSI
        # for the receiver as it only needs one channel realization per OFDM symbol
        a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
        a_freq = a_freq[...,:rg.num_ofdm_symbols*number_of_slots]

        # Compute the channel frequency response
        h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)

        h_hat, err_var = remove_nulled_scs(h_freq), 0.

        # Extract every 75th element from the 5th dimension
        ChannelTF_subbatch = h_hat[:, 0, :, 0,  :, ::75, 0]
        ChannelTF_subbatch = tf.transpose(ChannelTF_subbatch, perm=[0,3,1,2])
        ChannelNP_subbatch = ChannelTF_subbatch.numpy()
        
        if(subbatch == 0):
            ChannelNP_batch = ChannelNP_subbatch
        else:
            ChannelNP_batch = np.append(ChannelNP_batch, ChannelNP_subbatch, axis=0) 

    if(batch == 0):
        ChannelNP = np.expand_dims(ChannelNP_batch, axis=0)
    else:
        ChannelNP_batch = np.expand_dims(ChannelNP_batch, axis=0)
        ChannelNP = np.append(ChannelNP,ChannelNP_batch,axis=0)


    # Stack the real and imaginary parts along the last dimension
    
    ChannelPyTorch = torch.from_numpy(ChannelNP)

file = f'GeneratedChannels/ChannelCDL{cdl_model}_S{number_of_slots}_{num_bs_ant}_Rx{num_ut_ant}_DS{delay_spread}_V{int(speed*3.6)}_{direction}.pickle'

print(f"Saving data with shape {ChannelPyTorch.size}")

# Check if the file exists
try:
    with open(file, 'rb') as handle:
        existing_data = pickle.load(handle)
    # Append new data to the existing file
    ChannelPyTorch = np.concatenate((existing_data, ChannelPyTorch), axis=0)
    ChannelPyTorch = torch.from_numpy(ChannelPyTorch)
    print(ChannelPyTorch.shape)
    with open(file, 'wb') as handle:
        pickle.dump(ChannelPyTorch, handle)
    print("New data appended to the existing file.")
except FileNotFoundError:
    # If the file doesn't exist, create a new file
    with open(file, 'wb') as handle:
        pickle.dump(ChannelPyTorch, handle)
    print("File created with new data.")
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import tensorflow as tf

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


rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=76,
                  subcarrier_spacing=15e3,
                  num_tx=1,
                  num_streams_per_tx=num_streams_per_tx,        
                  cyclic_prefix_length=6,
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


delay_spread = 10e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value. 

direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.
cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]

speed = 30/3.6            # UT speed [m/s]. BSs are always assumed to be fixed.
                      # The direction of travel will chosen randomly within the x-y plane.

# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)

l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max-l_min+1

batch_size = 1
ebno_db = 30


# The following values for truncation are recommended.
# Please feel free to tailor them to you needs.

a, tau  = cdl(batch_size=batch_size, num_time_steps=rg.num_time_samples*20+l_tot-1, sampling_frequency=rg.bandwidth)

# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    
    
# We need to sub-sample the channel impulse reponse to compute perfect CSI
# for the receiver as it only needs one channel realization per OFDM symbol
a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
a_freq = a_freq[...,:rg.num_ofdm_symbols*20]

# Compute the channel frequency response
h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)

h_hat, err_var = remove_nulled_scs(h_freq), 0.





for i in range(num_bs_ant):
    plt.subplot(2,2,i+1)
    plt.plot(np.arange(280)/14,np.real(h_hat)[0,0,i,0,0,:,0])
    plt.stem(np.real(h_hat)[0,0,i,0,0,:,0][1::14])
    plt.plot(np.arange(280)/14,np.imag(h_hat)[0,0,i,0,0,:,0])
    plt.xlabel("OFDM symbol")
    plt.ylabel("Real part of the channel")
    plt.title(f"Effective channel between\n the 1 UE antenna and the \n{i+1} BS antenna")
plt.tight_layout()    
plt.show()
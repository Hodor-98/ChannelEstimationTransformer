
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
import os

import sionna
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


from models.model import Informer, InformerStack, LSTM, RNN, GRU, InformerStack_e2e


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

# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
sionna.config.xla_compat=True
class Model(tf.keras.Model):
    """

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.
        Time-domain simulations are generally slower and consume more memory.
        They allow modeling of inter-symbol interference and channel changes
        during the duration of an OFDM symbol.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use. Note that "D" and "E" are LOS models that are
        not well suited for the transmissions of multiple streams.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed. For downlink
        transmissions, the transmitter is always assumed to have perfect CSI.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of independent Mote Carlo simulations
        to be performed at once. The larger this number, the larger the memory
        requiremens.

    ebno_db : float
        The Eb/No [dB]. This value is converted to an equivalent noise power
        by taking the modulation order, coderate, pilot and OFDM-related
        overheads into account.

    Output
    ------
    b : [batch_size, 1, num_streams, k], tf.float32
        The tensor of transmitted information bits for each stream.

    b_hat : [batch_size, 1, num_streams, k], tf.float32
        The tensor of received information bits for each stream.
    """

    def __init__(self,
                 direction,
                 cdl_model,
                 channel_estimation,
                 delay_spread,
                 perfect_csi,
                 speed,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 15e3
                ):
        super().__init__()

        # Provided parameters
        self._direction = direction
        self._cdl_model = cdl_model
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 72
        self._num_ofdm_symbols = 14
        self._num_ut_ant = 4 # Must be a multiple of two as dual-polarized antennas are used
        self._num_bs_ant = 8 # Must be a multiple of two as dual-polarized antennas are used
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 2
        self._coderate = 0.5
        
        self._channel_estimation = channel_estimation
        
        # Initialize transformer
        if(self._channel_estimation == "Transformer"):
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
            
            use_gpu = False

            device = torch.device('cuda:0') if use_gpu else torch.device('cpu')   # Example value, replace this with your device choice
            self._device = device
            
            transformer = InformerStack(
                enc_in, dec_in, c_out, seq_len, label_len, pred_len, factor, d_model, n_heads,
                e_layers, d_layers, d_ff, dropout, attn, embed, activation, output_attention,
                distil, device
            )
            
            state_dict = torch.load(f"TrainedTransformers/{transformer.__class__.__name__}_best_model_params_V{speed}_{direction}.pt", map_location=torch.device('cpu'))
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            transformer.load_state_dict(state_dict)
            transformer = torch.nn.DataParallel( transformer ).cuda() if use_gpu else transformer 
            print("transformer has been loaded!")
            
            self._transformer = transformer

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        self._ut_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_ut_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)



        self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                l_tot=self._l_tot,
                                                add_awgn=True)
        self._modulator = OFDMModulator(self._cyclic_prefix_length)
        self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    @tf.function(jit_compile=True) # See the following guide: https://www.tensorflow.org/guide/function
    def call(self, batch_size, ebno_db):
        
        number_of_slots = 50

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        # Time-domain simulations
        a, tau = self._cdl(batch_size, self._rg.num_time_samples*number_of_slots+self._l_tot-1, self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                        l_min=self._l_min, l_max=self._l_max, normalize=True)

        # As precoding is done in the frequency domain, we need to downsample
        # the path gains `a` to the OFDM symbol rate prior to converting the CIR
        # to the channel frequency response.
        a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
        a_freq = a_freq[...,:self._rg.num_ofdm_symbols*number_of_slots]
        h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
        print(h_time.shape)
        print(h_freq.shape)

        if self._direction == "downlink":
            x_rg, g = self._zf_precoder([x_rg, h_freq])

        x_time = self._modulator(x_rg)
        y_time = self._channel_time([x_time, h_time, no])

        y = self._demodulator(y_time)

        if(self._channel_estimation == "Transformer"):
           
            # data = LoadBatch(h_freq[:, :25, :, :])

            # inp_net = data.to(self._device)

            # enc_inp = inp_net
            # dec_inp =  torch.zeros_like( enc_inp[:, -pred_len:, :] ).to(self._device)
            # dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)

            # # informer
            # outputs_informer = self._transformer(enc_inp, dec_inp)[0]
            
            print("temp")
           
        else:
            if self._perfect_csi:
                if self._direction == "uplink":
                    h_hat = self._remove_nulled_scs(h_freq)
                elif self._direction =="downlink":
                    h_hat = g
                err_var = 0.0
            else:
                h_hat, err_var = self._ls_est ([y, no])

        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)

        return b, b_hat


UL_SIMS = {
    "ebno_db" : list(np.arange(5, 11, 3.0)),
    "cdl_model" : ["B"],
    "delay_spread" : 100e-9,
    "direction" : "uplink",
    "perfect_csi" : True,
    "speed" : 30,
    "channel_estimation": "Transformere",
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for cdl_model in UL_SIMS["cdl_model"]:

    model = Model(direction=UL_SIMS["direction"],
                  cdl_model=cdl_model,
                  delay_spread=UL_SIMS["delay_spread"],
                  perfect_csi=UL_SIMS["perfect_csi"],
                  speed=UL_SIMS["speed"],
                  channel_estimation=UL_SIMS["channel_estimation"],
                  cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])

    ber, bler = sim_ber(model,
                        UL_SIMS["ebno_db"],
                        batch_size=32,
                        max_mc_iter=100,
                        num_target_block_errors=1000)

    UL_SIMS["ber"].append(list(ber.numpy()))
    UL_SIMS["bler"].append(list(bler.numpy()))

UL_SIMS["duration"] = time.time() - start
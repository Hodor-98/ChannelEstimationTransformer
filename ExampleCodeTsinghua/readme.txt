This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] H, Jiang, M. Cui, D. W. K. Ng, and L. Dai, "Accurate channel prediction based on Transformer: Making mobility negligible," IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2717-2732, Sep. 2022.

*********************************************************************************************************************************
If you use this simulation code package in any way, please cite the original paper [1] above. 
 
The author in charge of this simulation code pacakge is: Hao Jiang (email: jiang-h18@mails.tsinghua.edu.cn).

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers (more information can be found at: 
http://oa.ee.tsinghua.edu.cn/dailinglong/publications/publications.html). 

Please note that the pytorch 1.0.0 in Python3.6 is used for this simulation code package,  and there may be some imcompatibility problems among different python or pytorch versions. 

Copyright reserved by the Broadband Communications and Signal Processing Laboratory (led by Dr. Linglong Dai), Beijing National Research Center for Information Science and Technology (BNRist), Department of Electronic Engineering, Tsinghua University, Beijing 100084, China. 

*********************************************************************************************************************************

Abstract of the paper: 
Accurate channel prediction is vital to address the channel aging issue in mobile communications with fast
time-varying channels. Existing channel prediction schemes are generally based on the sequential signal processing, i.e., the
channel in next time slot can only be sequentially predicted. Thus, the accuracy of channel prediction rapidly degrades with
the evolution of time slot due to the error propagation problem in the sequential operation. To overcome this challenging problem,
we propose a transformer-based parallel channel prediction scheme to predict future channels in parallel. Specifically, we first
formulate the channel prediction problem as a parallel channel mapping problem, which predicts the channels in next several
time slots in parallel. Then, inspired by the recently proposed parallel vector mapping model named transformer, a transformerbased
parallel channel prediction scheme is proposed to solve this formulated problem. Relying on the attention mechanism
in machine learning, the transformer-based scheme naturally enables parallel signal processing to avoid the error propagation
problem. The transformer can also adaptively assign more weights and resources to the more relevant historical channels
to facilitate accurate prediction for future channels. Moreover, we propose a pilot-to-precoder (P2P) prediction scheme that
incorporates the transformer-based parallel channel prediction as well as pilot-based channel estimation and precoding. In this way,
the dedicated channel estimation and precoding can be avoided to reduce the signal processing complexity. Finally, simulation
results verify that the proposed schemes are able to achieve a negligible sum-rate performance loss for practical 5G systems in
mobile scenarios.

*********************************************************************************************************************************
How to use this simulation code package?

(1) Run "attention_sample.m", you will see Fig. 4  "A sample of the continuous channels in P + L time slots". 

(2) Run "plot_channel.m", you will see Fig. 8 "A parallel channel prediction results of the proposed transformer-based parallel channel prediction scheme".

(3) Run "plot_nmse.m", you will see Fig. 9 & 10 "The NMSE performance versus time slot at v = 30 & 60 km/h.".

(4) Run "plot_rate.m", you will see Fig. 11 & 12 "The achievable sum-rate performance versus time slot at v = 30 & 60 km/h.".

(5) Note that the transformer, RNN, and LSTM models are well trained in last simulation, and the trained model parameters are saved in the floder "prediciton_code\checkpoints\checkpoints_30-60_L5". 
    And then you can run the code called as "prediciton_code/test_transformer_lstm_AR" to get the NMSE & achievable sum-rate results.
    The NMSE & achievable sum-rate results are saved in the folder named "prediciton_code/results"

It is noted that there may be some differences in the results of different training processes. 
*********************************************************************************************************************************
Enjoy the reproducible research!
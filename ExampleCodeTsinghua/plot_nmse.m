clear all; close all;
speed = 60;
data = "data/";
NMSE_Ideal = load(data+"NMSE_Ideal_"+string(speed)+".mat");
NMSE_transformer = load(data+"NMSE_transformer_"+string(speed)+".mat");
NMSE_LSTM = load(data+"NMSE_LSTM_"+string(speed)+".mat");
NMSE_GRU = load(data+"NMSE_GRU_"+string(speed)+".mat");
NMSE_RNN = load(data+"NMSE_RNN_"+string(speed)+".mat");
NMSE_PVEC = load(data+"NMSE_PVEC_"+string(speed)+".mat");
NMSE_PAD = load(data+"NMSE_PAD_"+string(speed)+".mat");
NMSE_Previous = load(data+"NMSE_Previous_"+string(speed)+".mat");

x = 0:5;
figure; hold on; box on; grid on;
plot(x, 10*log10(NMSE_Previous.NMSE), 's-', 'Linewidth',1.6,'color',[0.47,0.67,0.19])
plot(x, 10*log10(NMSE_PAD.NMSE), '>-', 'Linewidth',1.6,'color',[0.93,0.69,0.13])
plot(x, 10*log10(NMSE_PVEC.NMSE), '<-', 'Linewidth',1.6,'color',[0.49,0.18,0.56])
plot(x, 10*log10(NMSE_RNN.NMSE), 'x-', 'Linewidth',1.6,'color',[0.85,0.33,0.10])
plot(x, 10*log10(NMSE_LSTM.NMSE), 'o-', 'Linewidth',1.6,'color',[0.00,0.45,0.74])
plot(x, 10*log10(NMSE_transformer.NMSE), 'r-^', 'Linewidth',1.6)
%plot(x, 10*log10(NMSE_GRU.NMSE), 'd-', 'Linewidth',1.6)
legend({'No prediction', 'Sequential PAD [6]', 'Sequential PVEC [6]' , 'Sequential RNN [11]','Sequential LSTM [12] ','Parallel Transformer',},'Interpreter','latex')

xlabel('Time slot (0.625 ms)')
%ylim([min(10*log10(NMSE_transformer.NMSE))-1, max(10*log10(NMSE_Previous.NMSE)) + 0.3])
set(gca,'XTick',1:1:5);
set(gca,'XTicklabel',{'1','2','3','4','5'})
grid on;
ylabel('NMSE (dB)')
%title(string(speed) + ' km/h')

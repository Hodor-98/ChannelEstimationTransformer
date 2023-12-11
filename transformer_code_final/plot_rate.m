speed =60;
data = "data/";
Rate_Ideal = load(data+"Rate_Ideal_"+string(speed)+".mat");
Rate_transformer = load(data+"Rate_transformer_"+string(speed)+".mat");
Rate_transformer_e2e = load(data+"Rate_transformer_e2e_"+string(speed)+".mat");
Rate_LSTM = load(data+"Rate_LSTM_"+string(speed)+".mat");
Rate_GRU = load(data+"Rate_GRU_"+string(speed)+".mat");
Rate_RNN = load(data+"Rate_RNN_"+string(speed)+".mat");
Rate_PVEC = load(data+"Rate_PVEC_"+string(speed)+".mat");
Rate_PAD = load(data+"Rate_PAD_"+string(speed)+".mat");
Rate_Previous = load(data+"Rate_Previous_"+string(speed)+".mat");
x = 0:5;
Rate_transformer_e2e.rate(1) = Rate_transformer.rate(1);
figure; hold on; box on; grid on;
plot(x, Rate_Ideal.rate, 'k-.', 'Linewidth',1.6)
%plot(x, Rate_transformer_e2e.rate, '-p', 'Linewidth',1.6,'color',[0.60,0.196080,0.8])
plot(x, Rate_transformer.rate, 'r-^', 'Linewidth',1.6)
plot(x, Rate_LSTM.rate, 'o-', 'Linewidth',1.6,'color',[0.00,0.45,0.74])
%plot(x, Rate_GRU.rate, 'd-', 'Linewidth',1.6)
plot(x, Rate_RNN.rate, 'x-', 'Linewidth',1.6,'color',[0.85,0.33,0.10])
plot(x, Rate_PVEC.rate, '<-', 'Linewidth',1.6,'color',[0.49,0.18,0.56])
plot(x, Rate_PAD.rate, '>-', 'Linewidth',1.6,'color',[0.93,0.69,0.13])
plot(x, Rate_Previous.rate, 's-', 'Linewidth',1.6,'color',[0.47,0.67,0.19])
legend({'Perfect CSI','Parallel Transformer', 'Sequential LSTM [12]', 'Sequential RNN [11]', 'Sequential PVER [6]', 'Sequential PAD [6]','No prediction'},'Interpreter','latex')
ylim([min(Rate_Previous.rate)-0.7, max(Rate_Ideal.rate) + 0.3])
xlabel('Time slot (0.625 ms)')
set(gca,'XTick',0:1:5);
set(gca,'XTicklabel',{'0','1','2','3','4','5'})
ylabel('Achievable sum-rate (bps/Hz)')
%title(string(speed) + ' km/h')


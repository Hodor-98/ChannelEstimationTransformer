clear all; close all;
speed = 30;
data = "data/";
transformer = load(data+"CR_transformer_"+string(speed)+".mat");
data= load(data+"CR_data_"+string(speed)+".mat");

k = 49;
N = size(data.data, 2);
N_pred = size(transformer.data, 2);
x = 1:N;
figure; 
hold on; box on; grid on;
plot(x(1:end), real(data.data(k,1:end,4,1)), 'o--', 'Linewidth',1.6)
plot(x(N-N_pred + 1 : N), real(transformer.data(k,:,4,1)), 'o-', 'Linewidth',1.6)
xlabel('Time slot (0.625 ms)')
ylabel('Real part of channel')
legend({'Real channel', 'Predicted channel'},'Interpreter','latex','FontSize',16)

subplot(2,2,1)
hold on; box on; grid on;
plot(x(1:end), real(data.data(k,1:end,1,1)), 'o--', 'Linewidth',1.6)
plot(x(N-N_pred + 1 : N), real(transformer.data(k,:,1,1)), 'o-', 'Linewidth',1.6)
xlabel('Time slot')
ylabel('Real part of channel')

subplot(2,2,2)
hold on; box on; grid on;
plot(x(1:end), real(data.data(k,1:end,2,1)), 'o--', 'Linewidth',1.6)
plot(x(N-N_pred + 1 : N), real(transformer.data(k,:,2,1)), 'o-', 'Linewidth',1.6)
xlabel('Time slot')
ylabel('Real part of channel')

subplot(2,2,3)
hold on; box on; grid on;
plot(x(1:end), real(data.data(k,1:end,3,1)), 'o--', 'Linewidth',1.6)
plot(x(N-N_pred + 1 : N), real(transformer.data(k,:,3,1)), 'o-', 'Linewidth',1.6)
xlabel('Time slot')
ylabel('Real part of channel')

subplot(2,2,4)
hold on; box on; grid on;
plot(x(1:end), real(data.data(k,1:end,4,1)), 'o--', 'Linewidth',1.6)
plot(x(N-N_pred + 1 : N), real(transformer.data(k,:,4,1)), 'o-', 'Linewidth',1.6)
xlabel('Time slot')
ylabel('Real part of channel')
% xlabel('0.625 ms')

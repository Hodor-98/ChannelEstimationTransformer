import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os 
import torch
import time 
import math
from scipy.fftpack import fft, ifft 
from scipy.interpolate import interp1d
from metrics import NMSELoss
pi = np.pi

# def get_rate(H, sigma2):
#     a1 = H[0,0]
#     a2 = H[1,1]
#     I1 = H[0,1]
#     I2 = H[1,0]
#     rate1 = np.log2( 1 + np.abs(a1)**2 / ( np.abs(I1)**2 + sigma2) )
#     rate2 = np.log2( 1 + np.abs(a2)**2 / ( np.abs(I2)**2 + sigma2) )
#     rate = rate1 + rate2
#     return rate 
def get_rate(H, sigma2):
    rate = np.log2(np.linalg.det(np.eye(2) + 1/sigma2 * H.T.conj().dot(H)))
    return rate

def get_zf_rate(H_hat, H_true, SNR):
    D_np = get_zf_precoder(H_hat)
    HF = np.matmul(D_np, H_true)
    rate = SNR_rate(HF, SNR)
    return rate

def get_zf_precoder(H_hat):
    D_np = np.linalg.pinv(H_hat) # shape = [64, 2, 4]
    D_np = D_np / np.linalg.norm(D_np, axis=(2), keepdims=True)
    return D_np

def SNR_rate(H, SNR):
    rate = np.mean(np.log2(np.linalg.det(np.eye(2) + (10**(SNR/10)) * np.matmul(H.conj().transpose(0,2,1), H))))
    return rate

def SINR_rate(HF, SNR):
    HF = torch.pow(torch.abs(torch.from_numpy(HF).cuda()), 2)
    HF_diag = HF * torch.eye(2).cuda()                
    rate = torch.mean(torch.sum(torch.log2(1 + torch.sum(HF_diag, 2)\
        /(torch.abs(torch.sum(HF - HF_diag, 2))+ 1/(10**(SNR/10)))),1))
    return rate

def interpolate(H_prev, H_pred, ir):
    M, pred_len, N = H_pred.shape 
    _, prev_len, N = H_prev.shape 
    H = np.concatenate([H_prev, H_pred],  1)
    x = np.arange((pred_len + prev_len - 1) * 5 + 1)
    x0 = np.arange(pred_len + prev_len) *  ir
    x1_1 = np.arange(prev_len) *  ir
    x1_2 = np.arange( (prev_len - 1) *  ir + 1, (prev_len + pred_len -1) *  ir + 1)
    x1 = np.concatenate([x1_1, x1_2])

    H_interp = np.zeros([M, x1.size, N], dtype = np.complex)
    for i in range(M):
        for j in range(N):
            f = interp1d(x0, H[i, :, j], kind = 'cubic')
            H_interp[i, :, j] = f(x1)

    # plt.figure()
    # plt.plot(x, data[0,:,0,0].real, '--')
    # plt.plot(x0, H[0,:,0].real, '+')
    # plt.plot(x1, H_interp[0,:,0].real)
    # plt.plot(x1_2, H_interp[0,- pred_len *  ir:,0].real)
    # plt.savefig('test.png')
    return H_interp[:, - pred_len *  ir:, :]

def complex2real(data):
    B, P, N = data.shape
    data2 = data.reshape([B, P, N, 2])
    data2[...,0] = data.real 
    data2[...,1] = data.imag    
    return data2

def real2complex(data):
    B, P, N = data.shape 
    data2 = data.reshape([B, P, N//2, 2])
    data2 = data2[:,:,:,0] + 1j * data2[:,:,:,1]
    return data2



def get_result(tensor, Nt = 4, Nr = 2):
    # tensor shape: Batch * seq_len * 2 * subcarrier * (Nt \times Nr)
    result = np.array(tensor)
    result = result[:,:,0,:,:] + 1j * result[:,:,0,:,:]
    shape = list(result.shape)[:-1] 
    shape.extend([Nr, Nt])
    # print(shape)
    result = result.reshape(shape)
    return result 


def Torch_Complex_Matrix_Matmul(A, B):
    Ar = A[:,:,:,:,0]
    Ai = A[:,:,:,:,1]
    Br = B[:,:,:,:,0]
    Bi = B[:,:,:,:,1]
    A1 = torch.cat([torch.cat([Ar,-Ai],3),torch.cat([Ai,Ar],3)],2)
    B1 = torch.cat([Br,Bi],2)
    C = torch.matmul(A1,B1)    
    C = torch.cat((C[:,:,:int(C.size(2)/2),:].unsqueeze(4), C[:,:,int(C.size(2)/2):,:].unsqueeze(4)),4)
    return C


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


def get2DDFT(Nx, Ny):
    az =  np.linspace(-1/2 + 1/Nx, 1/2, Nx).reshape(1,Nx)
    el =  np.linspace(-1/2 + 1/Ny, 1/2, Ny).reshape(1,Ny)
    A_az = np.exp(-1j * 2 * pi * (np.arange(Nx).reshape(Nx,1)).dot(az))
    A_el = np.exp(-1j * 2 * pi * (np.arange(Ny).reshape(Ny,1)).dot(el))
    A = np.kron(A_az, A_el)/np.sqrt(Nx*Ny)
    return A



# Check GPU memory
def check_gpu_memory(use_gpu):
    if (torch.cuda.is_available() and use_gpu):
        current_device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    else:
        print("No GPU available.")

# Force cuDNN initialization
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda:0')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

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




# Function to train the model using SeqData DataLoader
def train(model: torch.nn.Module, optimizer, scheduler, epoch, dataloader,device) -> None:
    model.train()
    total_loss = 0.
    criterion = NMSELoss()
    start_time = time.time()
    
    batch = next(iter(dataloader))
    for itr in range(dataloader.batch_size):
        H, H_seq, H_pred = [tensor[itr] for tensor in batch]
        print(device)
        data  = LoadBatch(H)
        data = data.to(device)
        
        output = model.train_data(data, device)
        loss = criterion(data[:,-15:,...], output[:,-15:,...])
        
        # outputs_plot_test = real2complex(np.array(output.detach().cpu()))
        # data_plot = real2complex(np.array(data.detach().cpu()))
        
        # if( itr < 10):
        #     plt.figure()
        #     plt.plot(data_plot[0,:,0])
        #     plt.plot(outputs_plot_test[0,:,0], linestyle='--')
        #     plt.savefig(f"ChannelPredictionPlots/Prediction_{model.__class__.__name__}_{itr}_temp.png", dpi=300)
        #     plt.close()
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()
        if itr   % (dataloader.batch_size // 8) == 0 and itr > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / (dataloader.batch_size // 8)
            cur_loss = total_loss / (dataloader.batch_size // 8)
            ppl = math.exp(cur_loss)

            print(f'| epoch {epoch:3d} | {itr:5d}/{dataloader.batch_size:5d} batches | '
                f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.4f} | ppl {ppl:8.2f}', flush=True)
            total_loss = 0
            start_time = time.time()

def evaluate(model, evaluaterLoader,device):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    criterion = NMSELoss()
    
    with torch.no_grad():
        batch = next(iter(evaluaterLoader))
                    
        for itr in range(evaluaterLoader.batch_size):
            H, H_seq, H_pred = [tensor[itr] for tensor in batch]
            
            data, label = LoadBatch(H_seq), LoadBatch(H_pred)
            label = label.to(device)
            enc_inp = data.to(device)


            output = model.test_data(enc_inp, H_pred.shape[1], device)
            
            total_loss =  criterion(label, output)

            outputs_plot_test = real2complex(np.array(output.detach().cpu()))
            
            # print(outputs_plot_test.shape)
            # print(H.shape)
                
            x = np.array(list(range(H.shape[1])))
            
            plt.figure()
            for i in range(4):
                plt.subplot(2,2,i+1)
                plt.plot(x[-H_pred.shape[1]:],outputs_plot_test[0,:,i*2].real)
                plt.plot(H[0,:,i,0].real, linestyle='--')
            plt.savefig(f"ChannelPredictionPlots/Prediction_{model.__class__.__name__}_{itr}.png", dpi=300)
            plt.close()
        
    return total_loss


# Function to run the training loop using SeqData DataLoader
def train_loop(model: torch.nn.Module, trainerLoader, evaluaterLoader, epochs, lr, model_dict_name,device):
    best_val_loss = float('inf')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, optimizer, scheduler, epoch, trainerLoader,device)
        val_loss = evaluate(model, evaluaterLoader,device)  # You should modify evaluate() to use SeqData DataLoader for validation as well
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89, flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dict_name)

        scheduler.step()




        
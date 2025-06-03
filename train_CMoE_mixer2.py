# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:25:28 2025

@author: 25900
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from MoE_mixer import *
from scipy.io import loadmat
import math
import random

import matplotlib.pyplot as plt
import scipy.io as sio # mat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def nmse_torch(pred, truth):
    # pred, truth: [B, N, 2] 或 [B, N] 视情况
    error = (pred - truth).pow(2).sum(dim=-1)  # [B, N], 每个天线的均方误差
    power = truth.pow(2).sum(dim=-1)          # [B, N], 每个天线的真实功率
    nmse_per_sample = error.sum(dim=1) / power.sum(dim=1)  # [B]
    return nmse_per_sample.mean().item()  # 标量



def create_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr, train_ratio=0.8, batch_size=256):
    """
    Creates a data loader for the given parameters.
    
    Args:
        num_wave (int): Number of waveguides.
        num_ants (int): Number of antennas.
        num_pilots (int): Number of pilots.
        num_samples (int): Number of samples.
        train_snr (int): Training SNR.
        train_ratio (float): Ratio of training data.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader.
    """
    # Generate data file name
    dataName = f'Pinching1_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'
    
    # Load data
    data = loadmat(dataName)
    inputs = data['inputs'].astype(np.float32)
    outputs = data['outputs'].astype(np.float32)
    
    # Convert to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    
    # Split dataset into training and testing sets
    num_train = int(train_ratio * len(inputs))
    train_inputs, train_outputs = inputs[:num_train], outputs[:num_train]
    test_inputs, test_outputs = inputs[num_train:], outputs[num_train:]
    
    # Create DataLoader
    train_dataset = TensorDataset(train_inputs, train_outputs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_inputs, test_outputs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

num_wave = 1
num_ants = 32
num_pilots = 8
num_samples = 100000
train_snr = 20

train_loader1, test_loader1 = create_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr)
num_ants = 24
train_loader2, test_loader2 = create_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr)
num_ants = 16
train_loader3, test_loader3 = create_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr)
num_ants = 8
train_loader4, test_loader4 = create_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr)

# train_loader, test_loader = create_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr)


# num_ants_ls = [16, 8, 4]  # 不同天线数
# loaders_train = []
# loaders_test  = []
# for na in num_ants_ls:
#     tr_loader, te_loader = create_data_loader(
#         num_wave, na, num_pilots, num_samples, train_snr,
#         train_ratio=0.8,
#         batch_size=256
#     )
#     loaders_train.append(tr_loader)
#     loaders_test.append(te_loader)



# 参数设置
pos_dim = 1      # 天线位置维度 (N)
sig_dim = num_pilots*4      # 导频信号维度 (real + imaginary)
embed_dim = 64   # 特征对齐维度
num_heads = 4    # 注意力头数
output_dim = 2   # 输出信道向量维度 (2N)

# 初始化模型

index_Net = 4

if index_Net == 1:
    model = SetBasedChannelEstimationModel(pos_dim=1, sig_dim=sig_dim, hidden_dim=128, num_heads=8, num_layers=4, output_dim=2).to(device)
    model_save = f'Transformer16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.pth'
    dataName_SNR = f'Transformer16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'
elif index_Net ==2:
    model = MLPChannelEstimationModel(pos_dim=1, sig_dim=sig_dim, hidden_dim=64, num_layers=4, output_dim=2).to(device)
    model_save = f'MLP16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.pth'
    dataName_SNR = f'MLP16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'
elif index_Net ==3:
    model = PASA(max_antennas=32, sig_dim=sig_dim, hidden_dim=64, mlp_dim=128, num_blocks=3, num_experts=1, output_dim=2).to(device)
    model_save = f'SA16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.pth'
    dataName_SNR = f'SA16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'

elif index_Net ==4:
    model = PADA(max_antennas=32, sig_dim=sig_dim, hidden_dim=64, mlp_dim=128, num_blocks=3, num_experts=1, output_dim=2).to(device)
    model_save = f'Mixer16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.pth'
    dataName_SNR = f'Mixer16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'

else:
    model = MultiModalChannelEstimationModel(max_antennas=32, sig_dim=sig_dim, hidden_dim=64, mlp_dim=128, num_blocks=3, num_experts=4, output_dim=2).to(device)
    model_save = f'MoE16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.pth'
    dataName_SNR = f'MoE16_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'


# model = MultiModalSetTransformerModel(pos_dim=1, sig_dim=8, hidden_dim=64, output_dim=2).to(device)

# model = torch.load(model_save).to(device)   

# model = MultiModalChannelEstimationModel(
#     sig_dim=sig_dim,
#     embed_dim=embed_dim,
#     num_heads=num_heads,
#     output_dim=output_dim
# ).to(device)



# import time
# execution_count= 1000

# flop_count = []
# params_count = []
# na = 16
# input1 = torch.randn(1,na,1).to(device)
# input2 = torch.randn(1,na,sig_dim).to(device)

# start_time = time.time()
# for i1 in range(execution_count):
#     # start_time = time.time()
#     Yhat1 = model(input1,input2)
#     # end_time = time.time()    
#     # total_time = end_time - start_time     

# end_time = time.time()    

# total_time = end_time - start_time     
# print(f"线性网络执行 {execution_count} 次的平均运行时间: {total_time} 秒")


# from thop import profile
# from thop import clever_format

# flops, params = profile(model, inputs=(input1,input2))
# flop_count.append(flops)
# params_count.append(params)
# flops, params = clever_format([flops, params], "%.3f")
# print('flops: ', flops, 'params: ', params)
    


# 数据加载
def split_inputs(batch_inputs):
    positions = batch_inputs[:, :,:pos_dim]        # 前 N 列为天线位置
    signals = batch_inputs[:, :,pos_dim:pos_dim+sig_dim] # 后 2 列为导频信号实虚部
    return positions, signals

# NMSE performance metric
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_C = x_real  + 1j * (x_imag )
    x_hat_C = x_hat_real  + 1j * (x_hat_imag )
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Initialize model, loss function, and optimizer
criterion = nn.L1Loss().to(device)  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 100
test_avg_min = 100

prob1 = 0.2
prob2 = 0.2
prob3 = 0.3

prob1 = 0.2
prob2 = 0.2



alpha = 0.01

# model = torch.load(model_save).to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # if torch.rand(1).item() < prob1:
    #     dataset = 1
    #     train_loader = train_loader1
    #     test_loader = test_loader1
    # elif torch.rand(1).item() > prob1 and torch.rand(1).item() < prob2:
    #     dataset = 2
    #     train_loader = train_loader2
    #     test_loader = test_loader2
    # elif torch.rand(1).item() > prob2 and torch.rand(1).item() < prob3:
    #     dataset = 3
    #     train_loader = train_loader3
    #     test_loader = test_loader3
    # else:
    #     dataset = 4
    #     train_loader = train_loader4
    #     test_loader = test_loader4
        
    # if torch.rand(1).item() < prob1:
    #     dataset = 1
    #     train_loader = train_loader1
    #     test_loader = test_loader1
    # # elif torch.rand(1).item() > prob1 and torch.rand(1).item() < prob2:
    # #     dataset = 2
    # #     train_loader = train_loader3
    # #     test_loader = test_loader3
    # else:
    #     dataset = 3
    #     train_loader = train_loader4
    #     test_loader = test_loader4
       
    # if prob_list is None:
    #     # 如果没有指定，就均分
    #     prob_list = [1.0 / len(train_loaders)] * len(train_loaders)

    # 为每个 train_loader 创建一个迭代器
    # loader_iters = [iter(tl) for tl in train_loaders]
    
    
    dataset = 1
    train_loader = train_loader3
    test_loader = test_loader3
    print(f'Selected dataset: {dataset:.4f}')
       
    
    for batch_inputs, batch_outputs in train_loader:
        
        lr = adjust_learning_rate(optimizer, epoch,1e-3,5e-5)
        
        
        
        # Zero the gradients
        
        batch_inputs = batch_inputs.to(device)
        batch_outputs = batch_outputs.to(device)
        positions, signals = split_inputs(batch_inputs)
        
        B, N = positions.shape[0], positions.shape[1]
        
        # positions = positions.to(device)
        # signals = signals.to(device)
        
        optimizer.zero_grad()
        predictions = model(positions, signals)

        # Forward pass
        loss = criterion(predictions, batch_outputs)
        
        # # 2) pick a random subset M < N
        # M = random.randint(1, N-1)  # or some rule
        # sub_idx = random.sample(range(N), M)  # subset of antenna indices
        # sub_idx = sorted(sub_idx)
        
        # # 3) extract the sub input (inp_sub) => shape [B, M, ...]
        # #    你需要保证 inp_full, out_full 里天线位置/信号对N维 -> sub_idx
        
        # inp_sub = batch_inputs[:, sub_idx, ...]   # => [B, M, ...]
        
        # # out_sub = out_full[:, sub_idx, :]     # => [B, M, 2]
        
        # positions, signals = split_inputs(inp_sub)
        
        # # inp_sub = inp_full[:, sub_idx, ...]   # => [B, M, ...]
        
        # # out_sub = out_full[:, sub_idx, :]     # => [B, M, 2]
        
        # pred_sub = model(positions, signals)
        
        # pinch_loss = pinching_consistency_loss(predictions, pred_sub, sub_idx)
        
        loss = loss  
        
        # Add EWC loss
        # if epoch > 0:
        #     ewc_loss.update_fisher(train_loader, model)
        #     loss += ewc_loss(model.named_parameters())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # ewc_loss.update_fisher(train_loader, model)

    # Testing loop
    model.eval()
    test_loss = 0.0
    tr_nmse1 = []
    with torch.no_grad():
        for batch_inputs, batch_outputs in test_loader:
            
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            positions, signals = split_inputs(batch_inputs)
            # positions = positions.cuda()
            # signals = signals.cuda()
            predictions = model(positions, signals)
            loss = criterion(predictions, batch_outputs)
            test_loss += loss.item()
            
            nmsei1=np.zeros([predictions.shape[0], 1])
            for i1 in range(predictions.shape[0]):
                nmsei1[i1] = np.sum(np.square(np.abs(predictions[i1,:].cpu().detach().numpy()-batch_outputs[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(batch_outputs[i1,:].cpu().detach().numpy())))
                # nmsei1[i1] = np.sum(np.square(np.abs(Yhat2[i1,:].cpu().detach().numpy()-YE2[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE2[i1,:].cpu().detach().numpy())))

                
            tr_nmse1.append(np.mean(nmsei1))
            
#        nm1.append(np.mean(tr_nmse1))
#        cost1D.append(epoch_cost1/len(val_loader))
        
        nmse_avg = 10*np.log10(np.mean(tr_nmse1))
        print('Iter-{}; NMSE_R1: {:.4}'.format(epoch, nmse_avg))
    
    # Print average test loss
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    
    if nmse_avg < test_avg_min:
        print('Model saved!')
        test_avg_min = nmse_avg

        torch.save(model, model_save)
#torch.save(model model_save)
        # torch.save(model.state_dict(), model_save)
        
        
def create_test_data_loader(num_wave, num_ants, num_pilots, num_samples, train_snr, batch_size=50):

    # Generate data file name
    dataName = f'PinchingTest1_{num_wave}waveguides_{num_ants}ants_{num_pilots}pilots_{num_samples}samples_{train_snr}snr.mat'
    
    # Load data
    data = loadmat(dataName)
    inputs = data['inputs'].astype(np.float32)
    outputs = data['outputs'].astype(np.float32)
    
    # Convert to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    
    test_dataset = TensorDataset(inputs, outputs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

test1_nmse=[]
nmse1_snr=[]

# num_ants = 6


model = torch.load(model_save).to(device)

# dataName = 'PinchingTest_{}waveguides_{}ants_{}pilots_{}samples_{}snr.mat'.format(num_wave, num_ants, num_pilots, num_samples, train_snr) 

# data = loadmat(dataName)

# # 提取输入和输出数据
# inputs = data['inputs'].astype(np.float32)
# outputs = data['outputs'].astype(np.float32)

# # Convert to PyTorch tensors
# inputs = torch.tensor(inputs, dtype=torch.float32)
# outputs = torch.tensor(outputs, dtype=torch.float32)

# # Create DataLoader
# batch_size = 50
# test_dataset = TensorDataset(inputs, outputs)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
start = 4
end = 33
step = 4
list_of_numbers = np.arange(start, end, step).tolist()
print(list_of_numbers)  # 输出: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
snrs = np.linspace(-20,20,9)

num_ants_ls = list_of_numbers  # 不同天线数
nmse1_db=np.zeros([len(list_of_numbers),len(snrs)])
i_ant = 0
for na in num_ants_ls:
    test_loader = create_test_data_loader(
        num_wave, na, num_pilots, num_samples, train_snr, batch_size=50
    )

    with torch.no_grad():
        model.eval()
        
        epoch_cost = 0
        tr_nmse1 = []
        for i, (batch_inputs, batch_outputs) in enumerate(test_loader): 
                    
            batch_inputs = batch_inputs.to(device)
            YE1 = batch_outputs.to(device)
            positions, signals = split_inputs(batch_inputs)
            # positions = positions.cuda()
            # signals = signals.cuda()
            Yhat1 = model(positions, signals)
            
            # nmsei1=np.zeros([YE1.shape[0], YE1.shape[1]])
            # for i1 in range(YE1.shape[0]):
            #     for i2 in range(YE1.shape[1]):
            #         nmsei1[i1,i2] = np.sum(np.square(np.abs(Yhat1[i1,i2,:].cpu().detach().numpy()-YE1[i1,i2,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,i2,:].cpu().detach().numpy())))
            # nmse1 =np.mean(nmsei1,axis=0)
            
            nmsei1=np.zeros([Yhat1.shape[0], 1])
            for i1 in range(Yhat1.shape[0]):
                nmsei1[i1] = np.sum(np.square(np.abs(Yhat1[i1,:].cpu().detach().numpy()-YE1[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:].cpu().detach().numpy())))
            nmse1 =np.mean(nmsei1,axis=0)
            
            
            # nmse1[29:31]=nmse1[20]
            # nmse1[30]=nmse1[20]
            
            test1_nmse.append(nmse1)
            if (i+1)%10==0:
                nmse1_snr.append(np.mean(test1_nmse,axis=0))
                test1_nmse=[]
    
    # 绘制NMSE结果图
    # NMSE v.s. SNR
    nmse1_db[i_ant,:]=10*np.log10(np.mean(nmse1_snr,axis=1))
    nmse1_snr = []
    i_ant = i_ant + 1


# NMSE v.s. SNR
plt.plot(snrs, nmse1_db[3,:],ls='-', marker='+', c='black',label='Linear')
plt.legend()
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()

# NMSE v.s. Antennas
plt.figure
# nmse1_db_s=10*np.log10(nmse1_snr[4])
slots = np.arange(start, end, step)
plt.plot(slots, nmse1_db[:,5],ls='-', marker='+', c='black',label='LSTM')
plt.legend()
plt.grid(True) 
plt.xlabel('Antennas')
plt.ylabel('NMSE/dB')
plt.show()

sio.savemat(dataName_SNR, {'a':nmse1_db})


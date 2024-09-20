# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:25:49 2020

@author: wb
"""
import torch
import torch.nn as nn
from ST_Transformer import STTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data, PrepareDataset
import scipy.sparse as sp
from IGNN.utils import sparse_mx_to_torch_sparse_tensor

from config import dataset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    days = 21       #选择训练的天数
    val_days = 3    #选择验证的天数
    tst_days = 6    #选择测试的天数

    train_num = 288*days
    val_num = 288*val_days
    tst_num = 288*tst_days
    row_num = train_num + val_num
    # 模型参数
    in_channels = 1  # 输入通道数*batch_size。只有速度信息，所以通道为1
    embed_size = 16  # Transformer通道数
    time_num = 288  # 1天时间间隔数量
    num_layers = 1  # Spatial-temporal block 堆叠层数
    T_dim = 12  # 输入时间维度。 输入前1小时数据，所以 60min/5min = 12
    output_T_dim = 12  # 输出时间维度。预测未来15,30,45min速度
    heads = 1  # transformer head 数量。 时、空transformer头数量相同

    # v, A = load_data(dataset)
    train_dataloader, valid_dataloader, test_dataloader, A, max_speed = PrepareDataset(dataset, train_num, val_num, tst_num, BATCH_SIZE=in_channels)

    
    # model input shape: [1, N, T]   
    # 1:初始通道数, N:传感器数量, T:时间数量
    # model output shape: [N, T]    
    model = STTransformer(
        A,
        dataset,
        in_channels, 
        embed_size, 
        time_num, 
        num_layers, 
        T_dim, 
        output_T_dim, 
        heads
    ).to(device)
    
    # optimizer, lr, loss按论文要求
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    ###################### print model parameter states ######################
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    ###########################################################################
    #   ----训练部分----
    # t表示遍历到的具体时间

    pltx, plt_mae, plt_mape, plt_rmse = [], [], [], []
    best_loss = np.inf
    for epoch in range(100):
        losses = []
        model.train()
        t = 0
        for data in train_dataloader:
            x, y = data
            x = x.to(device) # x shape:[B, N, T_dim]
            y = y.to(device) # y shape:[B, N, output_T_dim]

            out = model(x, t)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            t += 1
        print(epoch, "Train MAE loss:", loss)


        model.eval()
        t = 0
        for data in valid_dataloader:
            x, y = data
            x = x.to(device) # x shape:[1, N, T_dim]
            y = y.to(device) # y shape:[N, output_T_dim]

            out = model(x, t)
            loss = criterion(out, y)

            losses.append(loss.detach().cpu().numpy())
            t += 1

        if best_loss > np.mean(losses):
            best_loss = np.mean(losses)
            print(epoch, "--------------------------------- Best Val MAE loss:", best_loss)
            torch.save(model.state_dict(), "model_{}.pth".format(dataset))

    model.load_state_dict(torch.load("model_{}.pth".format(dataset)))
    model.eval()
    t = 0
    for data in test_dataloader:
        x, y = data
        x = x.to(device) # x shape:[1, N, T_dim]
        y = y.to(device) # y shape:[N, output_T_dim]

        out = model(x, t)
        loss = criterion(out, y)
        mape = torch.mean(torch.abs((y - out) / y))
        rmse = torch.sqrt(torch.mean((out - y)**2))

        pltx.append(t)
        plt_mae.append(loss.detach().cpu().numpy())
        plt_mape.append(mape.detach().cpu().numpy())
        plt_rmse.append(rmse.detach().cpu().numpy())
        t += 1

    print('Test MAE/MAPE/RMSE:', np.mean(plt_mae), np.mean(plt_mape), np.mean(plt_rmse))

    plt.plot(pltx, plt_mae, label="STTN MAE")
    plt.plot(pltx, plt_mape, label="STTN MAPE")
    plt.plot(pltx, plt_rmse, label="STTN RMSE")
    plt.title("ST-Transformer Test")
    plt.xlabel("t")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    
    
    
    
    
    
    
    
    
    
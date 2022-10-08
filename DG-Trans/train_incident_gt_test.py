# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:25:49 2020

@author: wb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from GT_Transformer_incident_test import STTransformer
import pandas as pd
import numpy as np
import random
from load_incident import load_incident
import scipy.sparse as sp

from config import dataset

torch.manual_seed(2)
torch.cuda.manual_seed(2)
np.random.seed(2)
random.seed(2)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


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
    batch_size = 1  # 输入通道数*batch_size。只有速度信息，所以通道为1
    in_channel = 2
    embed_size = 16  # Transformer通道数
    time_num = 2016  # 1天时间间隔数量
    num_layers = 1  # Spatial-temporal block 堆叠层数
    T_dim = 6  # 输入时间维度。 输入前1小时数据，所以 60min/5min = 12
    T_dim_af = 3  # number of timestamps after the incident
    heads = 2  # transformer head 数量。 时、空transformer头数量相同

    # v, A = load_data(dataset)
    dataloader = load_incident(dataset, batch_size, batch_size, batch_size)
    scaler = dataloader['scaler']
    A = dataloader['A'].float().to(device)
    adj_sr, adj_rr = dataloader['A_sr'], dataloader['A_rr']
    adj_sr, adj_rr = torch.from_numpy(adj_sr).float().to(device), torch.from_numpy(adj_rr).float().to(device)
    sensor_dist = torch.from_numpy(dataloader['sensor_dist']).float().to(device)
    # model input shape: [2, N, T]
    # 1:初始通道数, N:传感器数量, T:时间数量
    # model output shape: [N, T]    
    model = STTransformer(
        [A, adj_rr, adj_sr],
        sensor_dist,
        dataset,
        in_channel,
        embed_size, 
        time_num, 
        num_layers, 
        T_dim,
        T_dim_af,
        heads
    ).to(device)
    
    # optimizer, lr, loss按论文要求
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005, weight_decay=0.0)
    criterion = nn.L1Loss()
    ###################### print model parameter states ######################
    print('Net\'s state_dict:')
    total_param, total_layer = 0, 0
    for param_tensor in model.state_dict():
        print(total_layer, param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_layer += 1
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    ###########################################################################
    #   ----训练部分----
    # t表示遍历到的具体时间

    # torch.autograd.set_detect_anomaly(True)
    best_loss, best_train = np.inf, np.inf
    for epoch in range(100):
        losses, preds = [], []
        model.train()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).float().to(device) # x shape:[1, T_dim, N, D]
            x = x.permute(3, 2, 1, 0).squeeze()[:, :, 9:15]
            y = torch.Tensor(y).float().to(device) # y shape:[N, output_dim]
            # timestamp index, road index, category indx, distance of the road
            t, r, c, d = y[:, 0].round().type(torch.int).item(), y[:, 1].round().type(torch.int).item(), y[:, 2].round().type(torch.int).item(), y[:, 3].item()
            out, rr, cc, residual = model(x, r, d, t)
            loss1 = criterion(out, y[:, -2:])
            loss = loss1
            if epoch>20:
                optimizer.param_groups[0]['lr'] = 0.001
                plen = len([param for param in model.parameters()])
                others = plen - 94 # number of layers to be further trained
                for param in model.parameters():
                    param.requires_grad = False  # or True
                    plen -= 1
                    if plen == 30:
                        break
                r_label, c_label = torch.zeros_like(rr), torch.zeros_like(cc)
                r_label[:, r], c_label[:, c] = 1, 1
                loss3 = F.cross_entropy(cc, c_label)#+F.cross_entropy(rr, r_label) #criterion(residual, torch.zeros_like(residual).to(device))
                loss = loss1 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds.append(out.detach().cpu().numpy())

        if best_train > loss1:
            torch.save(model.state_dict(), "model_{}_GT_train_test.pth".format(dataset))
            preds = np.concatenate(preds)
            np.save(dataset + 'preds_GT_train_test', preds, allow_pickle=True)
            best_train = loss1

        print(epoch, "Train MAE loss:", loss1.item(), loss.item())


        model.eval()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x = torch.Tensor(x).float().to(device)
            x = x.permute(3, 2, 1, 0).squeeze()[:, :, 9:15]
            y = torch.Tensor(y).float().to(device)
            t, r, c, d = y[:, 0].round().type(torch.int).item(), y[:, 1].round().type(torch.int).item(), y[:, 2].round().type(torch.int).item(), y[:, 3].item()
            out, rr, cc, residual = model(x, r, d, t)
            loss = criterion(out, y[:, -2:])

            losses.append(loss.detach().cpu().numpy())

        if best_loss > np.mean(losses):
            best_loss = np.mean(losses)
            print(epoch, "--------------------------------- Best Val MAE loss:", best_loss)
            torch.save(model.state_dict(), "model_{}_GT_test.pth".format(dataset))

    model.load_state_dict(torch.load("model_{}_GT_test.pth".format(dataset)))
    model.eval()
    pltx, plt_mae, plt_mape, plt_rmse = [], [], [], []
    preds = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x = torch.Tensor(x).float().to(device)
        x = x.permute(3, 2, 1, 0).squeeze()[:, :, 9:15]
        y = torch.Tensor(y).float().to(device)
        t, r, c, d = y[:, 0].round().type(torch.int).item(), y[:, 1].round().type(torch.int).item(), y[:, 2].round().type(torch.int).item(), y[:, 3].item()
        out, rr, cc, residual = model(x, r, d, t)
        out, y = out.detach().cpu().numpy(), y.detach().cpu().numpy()
        loss = np.array([np.abs(out[:,0]-y[:,-2]), np.abs(out[:,1]-y[:,-1])])
        tmp = (y[:, -2:]-out)/(y[:, -2:]+out)*2
        mape = np.array([np.abs(tmp[:, 0]), np.abs(tmp[:,1])])
        rmse = np.array([(out[:,0] - y[:,-2])**2, (out[:,1] - y[:,-1])**2])

        pltx.append(t)
        preds.append(out)
        plt_mae.append(loss)
        plt_mape.append(mape)
        plt_rmse.append(rmse)

    plt_mae, plt_mape, plt_rmse = np.array(plt_mae), np.array(plt_mape), np.array(plt_rmse)
    print('MAE:', np.mean(plt_mae[:,0])*scaler[0], np.mean(plt_mae[:,1])*scaler[1], 'MAPE:', np.mean(plt_mape[:,0]), np.mean(plt_mape[:1]),
          'RMSE:', np.sqrt(np.mean(plt_rmse[:,0]))*scaler[0], np.sqrt(np.mean(plt_rmse[:,1]))*scaler[1])

    preds = np.concatenate(preds)
    np.save(dataset+'preds_GT_test', preds, allow_pickle=True)

    
    
    
    
    
    
    
    
    
    
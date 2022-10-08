# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:25:49 2020

@author: wb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from GT_Transformer_incident_best import STTransformer#, exe_time
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
    embed_size = 16  # hidden channel
    time_num = 18  # same as t_dim in this case
    num_layers = 1  # Spatial-temporal block 堆叠层数
    T_dim = 18  # 输入时间维度。 输入前1小时数据，所以 60min/5min = 12
    output_T_dim = 18  # 输出时间维度。预测未来15,30,45min速度
    heads = 1  # transformer head 数量。 时、空transformer头数量相同

    # v, A = load_data(dataset)
    dataloader = load_incident(dataset, batch_size, batch_size, batch_size)
    scaler = dataloader['scaler']
    A = dataloader['A'].float().to(device)
    adj_rr = pd.read_csv('../graphtransformer-main/data/'+dataset+'/adj_rr.csv', index_col=0, header=0).to_numpy()
    adj_sr = pd.read_csv('../graphtransformer-main/data/'+dataset+'/adj_sr.csv', index_col=0, header=0).to_numpy()
    adj_rr = adj_rr + np.eye(adj_rr.shape[0])
    adj_sr, adj_rr = torch.from_numpy(adj_sr).float().to(device), torch.from_numpy(adj_rr).float().to(device)
    sensor_dists = torch.from_numpy(dataloader['sensor_dist']).float().to(device)
    # model input shape: [2, N, T]
    # 1:初始通道数, N:传感器数量, T:时间数量
    # model output shape: [N, T]    
    model = STTransformer(
        [A, adj_rr, adj_sr, sensor_dists],
        dataset,
        in_channel,
        embed_size, 
        time_num, 
        num_layers, 
        T_dim, 
        output_T_dim, 
        heads
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=0.0) # best is 0.001
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
    #   ----training----
    # torch.autograd.set_detect_anomaly(True)
    best_loss, best_train = np.inf, np.inf
    for epoch in range(40):
        losses, preds = [], []
        model.train()
        t = 0
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).float().to(device) # x shape:[1, T_dim, N, D]
            x = x.permute(3, 2, 1, 0).squeeze()
            y = torch.Tensor(y).float().to(device) # y shape:[N, output_dim]

            out, c, residual = model(x, t)
            # att = att.permute(3, 2, 1, 0).reshape(-1, att.shape[0])
            w = 1/(t+1)
            tgt = x[:, :, 12:].permute(1, 2, 0)#.repeat(1,1,2)
            loss1 = criterion(out, y[:, -2:])
            loss2 = w*criterion(residual[0], tgt)+(1-w)*criterion(residual[1], tgt)#criterion(residual, torch.zeros_like(residual).to(device))
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds.append(out.detach().cpu().numpy())
            t += 1

        # print('time', np.mean(exe_time)/1e3)
        if best_train > loss1:
            torch.save(model.state_dict(), "model_{}_GT_train_2.pth".format(dataset))
            preds = np.concatenate(preds)
            np.save(dataset + 'preds_GT_train_2', preds, allow_pickle=True)
            best_train = loss1

        print(epoch, "Train MAE loss:", loss1, loss)


        model.eval()
        t = 0
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x = torch.Tensor(x).float().to(device)
            x = x.permute(3, 2, 1, 0).squeeze()
            y = torch.Tensor(y).float().to(device)

            out,c,_ = model(x, t)
            loss = criterion(out, y[:, -2:])

            losses.append(loss.detach().cpu().numpy())
            t += 1

        if best_loss > np.mean(losses):
            best_loss = np.mean(losses)
            print(epoch, "--------------------------------- Best Val MAE loss:", best_loss)
            torch.save(model.state_dict(), "model_{}_GT_2.pth".format(dataset))

    model.load_state_dict(torch.load("model_{}_GT_2.pth".format(dataset)))
    model.eval()
    t = 0
    pltx, plt_mae, plt_mape, plt_rmse = [], [], [], []
    preds, scores = [], []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x = torch.Tensor(x).float().to(device)
        x = x.permute(3, 2, 1, 0).squeeze()

        out,c,_ = model(x, t)
        out = out.detach().cpu().numpy()

        scores.append(c.detach().cpu().numpy())

        loss = np.array([np.abs(out[:,0]-y[:,-2]), np.abs(out[:,1]-y[:,-1])])
        tmp = (y[:, -2:]-out)/(y[:, -2:]+out)*2
        mape = np.array([np.abs(tmp[:, 0]), np.abs(tmp[:,1])])
        rmse = np.array([(out[:,0] - y[:,-2])**2, (out[:,1] - y[:,-1])**2])

        pltx.append(t)
        preds.append(out)
        plt_mae.append(loss)
        plt_mape.append(mape)
        plt_rmse.append(rmse)
        t += 1

    plt_mae, plt_mape, plt_rmse = np.array(plt_mae), np.array(plt_mape), np.array(plt_rmse)
    print('MAE:', np.mean(plt_mae[:,0])*scaler[0], np.mean(plt_mae[:,1])*scaler[1], 'MAPE:', np.mean(plt_mape[:,0]), np.mean(plt_mape[:1]),
          'RMSE:', np.sqrt(np.mean(plt_rmse[:,0]))*scaler[0], np.sqrt(np.mean(plt_rmse[:,1]))*scaler[1])

    preds = np.concatenate(preds)
    np.save(dataset + 'score_GT_2', scores, allow_pickle=True)
    np.save(dataset+'preds_GT_2', preds, allow_pickle=True)

    
    
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:25:49 2020

@author: wb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from GT_Transformer_incident_best import STTransformer
import pandas as pd
import numpy as np
import random
from load_incident import load_incident
from utils import SMAPE_np, SMAPE_torch, R2_np, MAE_np, RMSE_np, BMCLoss
from utils import wmae_loss as criterion_w
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
    batch_size = 8  # 输入通道数*batch_size。只有速度信息，所以通道为1
    in_channel = 2
    embed_size = 16  # hidden channel
    time_num = 9  # same as t_dim in this case
    num_layers = 1  # Spatial-temporal block 堆叠层数
    T_dim = 9  # 输入时间维度。 输入前1小时数据，所以 60min/5min = 12
    output_T_dim = 9  # 输出时间维度。预测未来15,30,45min速度
    heads = 4  # transformer head 数量。 时、空transformer头数量相同

    # v, A = load_data(dataset)
    dataloader = load_incident(dataset, batch_size, batch_size, batch_size)
    scaler = dataloader['scaler']
    adj_rr = pd.read_csv('../graphtransformer-main/data/'+dataset+'/adj_rr.csv', index_col=0, header=0).to_numpy()
    adj_sr = pd.read_csv('../graphtransformer-main/data/'+dataset+'/adj_sr.csv', index_col=0, header=0).to_numpy()
    adj_rr = adj_rr + np.eye(adj_rr.shape[0])
    adj_sr, adj_rr = torch.from_numpy(adj_sr).float().to(device), torch.from_numpy(adj_rr).float().to(device)
    # model input shape: [2, N, T]
    # 1:初始通道数, N:传感器数量, T:时间数量
    # model output shape: [N, T]    
    model = STTransformer(
        [adj_rr, adj_sr],
        dataset,
        batch_size,
        in_channel,
        embed_size, 
        time_num, 
        num_layers, 
        T_dim, 
        output_T_dim, 
        heads,
        device
    ).to(device)
    # model.load_state_dict(torch.load("model_{}_GT_5.pth".format(dataset)))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005, weight_decay=0.001) # best is 0.001
    criterion_L1 = nn.L1Loss()
    # init_noise_sigma, sigma_lr = 0.01, 0.001
    # criterion = BMCLoss(init_noise_sigma)
    # optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})
    ###################### print model parameter states ######################
    # print('Net\'s state_dict:')
    # total_param = 0
    # for param_tensor in model.state_dict():
    #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    #     total_param += np.prod(model.state_dict()[param_tensor].size())
    # print('Net\'s total params:', total_param)
    #
    # print('Optimizer\'s state_dict:')
    # for var_name in optimizer.state_dict():
    #     print(var_name, '\t', optimizer.state_dict()[var_name])
    ###########################################################################
    #   ----training----
    # torch.autograd.set_detect_anomaly(True)
    best_loss, best_train = np.inf, np.inf
    for epoch in range(20):
        losses, preds = [], []
        model.train()
        t = 0
        for iter, data in enumerate(dataloader['train_loader'].get_iterator()):
            x, y = data[0], data[1]
            x = torch.Tensor(x).float().to(device) # x shape:[B, T_dim, N, D]
            x = x.transpose(1, 3)
            y = torch.Tensor(y[:, -2:]).float().to(device) # y shape:[B, output_dim]
            out1, s1, adj1, residual1 = model(x, t, adj_sr, adj_rr)
            w = 1/(t+1)
            tgt = x[:, :, :, 6:].permute(0, 2, 3, 1)#.repeat(1,1,2)
            loss1 = criterion_L1(out1, y[:, -2:])#criterion(out1, y[:, -2:])
            # loss1 = SMAPE_torch(y[:, -2:], out) # SMAPE loss
            loss2 = w*criterion_L1(residual1[0], tgt)+(1-w)*criterion_L1(residual1[1], tgt)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(np.array([loss1.detach().cpu().numpy(), loss.detach().cpu().numpy()]))
            t += 1
        losses = np.array(losses)
        # print('time', np.mean(exe_time)/1e3)

        print(epoch, "Train MAE loss:", np.mean(losses[:, 0]), np.mean(losses[:, 1]))#, 'roads:', torch.sum(torch.abs(adj1[3]-model.Transformer.STransformer.R[None, :, :, :])).item())


        model.eval()
        t = 0
        losses = []
        for iter, data in enumerate(dataloader['val_loader'].get_iterator()):
            x, y = data[0], data[1]
            x = torch.Tensor(x).float().to(device)
            x = x.transpose(1, 3)
            rid = torch.Tensor(y[:, 2]).float().to(device)
            y = torch.Tensor(y[:, -2:]).float().to(device) # y shape:[B, output_dim]

            out,s1, a,_ = model(x, t, adj_sr, adj_rr)
            loss = criterion_L1(out1, y[:, -2:])#criterion(out, y[:, -2:])
            # smape = SMAPE_torch(y[:, -2:], out) # SMAPE loss

            losses.append(loss.detach().cpu().numpy())
            t += 1

        if best_loss > np.mean(losses):
            best_loss = np.mean(losses)
            print(epoch, "--------------------------------- Best Val MAE loss:", best_loss.item())
            torch.save(model.state_dict(), "model_{}_GT_nroad.pth".format(dataset))

    model.load_state_dict(torch.load("model_{}_GT_nroad.pth".format(dataset)))
    model.eval()
    t = 0
    preds, scores = [], []
    # adjs = []
    asrs, arrs, arss = [], [], []
    roads = []
    for iter, data in enumerate(dataloader['test_loader'].get_iterator()):
        x, y = data[0], data[1]
        x = torch.Tensor(x).float().to(device)
        x = x.transpose(1, 3)
        rid = torch.Tensor(y[:, 2]).float().to(device)
        y = torch.Tensor(y[:, -2:]).float().to(device) # y shape:[B, output_dim]

        out,s1, adj,_ = model(x, t, adj_sr, adj_rr)
        preds.append(out.detach().cpu().numpy())

        # scores.append(s1.detach().cpu().numpy())
        # adjs.append(adj[:, :, :, 6].mean(dim=-1).squeeze().detach().cpu().numpy())
        # asrs.append(adj[0][:, :, :, 6:].squeeze().detach().cpu().numpy())
        # arrs.append(adj[1][:, :, :, 6:].squeeze().detach().cpu().numpy())
        # arss.append(adj[2][:, :, :, 6:].squeeze().detach().cpu().numpy())
        # roads.append(adj[3].detach().cpu().numpy())
        t += 1

    preds = np.concatenate(preds)
    preds = preds[:dataloader['y_test'].shape[0]]
    y0, y1 = scaler[0].inverse_transform(dataloader['y_test'][:,-2]), scaler[1].inverse_transform(dataloader['y_test'][:,-1])
    p0, p1 = scaler[0].inverse_transform(preds[:,-2]), scaler[1].inverse_transform(preds[:,-1])

    print('MAE:', MAE_np(y0, p0),'MAPE:', SMAPE_np(y0, p0),
          'RMSE:', RMSE_np(y0, p0), 'R2:', R2_np(y0, p0))
    print('MAE:', MAE_np(y1, p1), 'MAPE:', SMAPE_np(y1, p1),
          'RMSE:', RMSE_np(y1, p1), 'R2:', R2_np(y1, p1))

    # preds, scores = np.concatenate(preds), np.concatenate(scores)
    # adjs = np.concatenate(adjs)
    # print(adjs.shape, model.Transformer.STransformer.I.shape)
    # asrs, arrs, arss = np.concatenate(asrs), np.concatenate(arrs), np.concatenate(arss)
    # roads = np.concatenate(roads)
    # np.save(dataset + 'score_GT_nroad', scores, allow_pickle=True)
    # np.save(dataset+'preds_GT_best_batch_test', preds, allow_pickle=True)
    # np.save(dataset+'adjs_GT_5', adjs, allow_pickle=True)
    # np.save(dataset+'asrs_GT_nroad', asrs, allow_pickle=True)
    # np.save(dataset+'arrs_GT_nroad', arrs, allow_pickle=True)
    # np.save(dataset+'arss_GT_nroad', arss, allow_pickle=True)
    # np.save(dataset+'R_init_nroad', model.Transformer.STransformer.R.squeeze().detach().cpu().numpy(), allow_pickle=True)
    # np.save(dataset+'R_GT_nroad', roads, allow_pickle=True)
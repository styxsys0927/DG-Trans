import torch
import numpy as np
import pandas as pd
import random
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import ChebConv
from layers.attentions import GAT_layer, TA_layer
from utils import SMAPE_np, SMAPE_torch, R2_np, MAE_np, RMSE_np
from load_incident import load_incident

exe_time=[]

torch.manual_seed(2)
torch.cuda.manual_seed(2)
np.random.seed(2)
random.seed(2)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

batch_size = 8
dataset = 'PEMS-07'
device = torch.device("cuda")

dataloader = load_incident(dataset, batch_size, batch_size, batch_size)
scaler = dataloader['scaler']
A = dataloader['A'].float().to(device)
adj_sr, adj_rr = torch.from_numpy(dataloader['A_sr']).float().to(device), torch.from_numpy(dataloader['A_rr']).float().to(device)
gt_m0, gt_m1 = scaler[0], scaler[1]

W = A - torch.eye(A.shape[0]).to(device)
eind_ss = W.nonzero().t().contiguous()
eind_rr = adj_rr.nonzero().t().contiguous()

# Graph transformer
class HastGCN(nn.Module):
    def __init__(self, N, Nr, T, T_minus, d_model, hidden_c=16, out_c=2):
        super(HastGCN, self).__init__()
        self.T_minus = T_minus
        self.start_conv = nn.Conv2d(d_model, hidden_c, (1,1))
        self.gat_s = GAT_layer(N, T, hidden_c)
        self.gcn_s = ChebConv(hidden_c, hidden_c, 3, bias=False)
        self.ta_minus = TA_layer(N, T_minus, hidden_c)
        self.ta_plus = TA_layer(N, T-T_minus, hidden_c)
        self.wq = nn.Parameter(torch.zeros(T-T_minus))
        self.wf = nn.Parameter(torch.zeros(hidden_c))
        self.wn = nn.Parameter(torch.zeros(N))
        self.gat_r = GAT_layer(Nr, T, hidden_c)
        self.gcn_r = ChebConv(hidden_c, hidden_c, 3)
        self.conv = nn.Conv1d(hidden_c, d_model, Nr*T)
        self.act = nn.ReLU()

    def forward(self, data, adj_rr, adj_sr, W, eind_ss, eind_rr):
        flow_x = self.act(self.start_conv(data.transpose(1, 3))).transpose(1, 3)  # (B, T, N, D)
        B, T, N, D = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        flow_x = self.gat_s(flow_x, W, mask=True)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        exe_time.append(start.elapsed_time(end))  # milliseconds

        flow_x = self.gcn_s(flow_x, eind_ss)
        tm_x = self.ta_minus(flow_x[:, :self.T_minus])
        tp_x = self.ta_minus(flow_x[:, self.T_minus:])
        gamma = torch.einsum('t,btnd->bnd', self.wq, tp_x)
        gamma = torch.einsum('bnd,d->bn', gamma, self.wf)
        gamma = torch.einsum('bn,n->b', gamma, self.wn)
        gamma = torch.sigmoid(gamma)
        out_tm, out_tp = (torch.ones(B).to(device)-gamma)[:, None, None, None]*tm_x, gamma[:, None, None, None]*tp_x
        output_x = torch.cat([out_tm, out_tp], dim=1) # (B, T, N, D)
        output_x = self.act(output_x)
        output_r = self.act(torch.einsum('btnd,nr->btrd', output_x, adj_sr)) # not sure
        output_r = self.gat_r(output_r, adj_rr)
        output_r = self.gcn_r(output_r, eind_rr)
        output_r = output_r.reshape(B, -1, D).permute(0, 2, 1)
        output = self.conv(output_r).squeeze(-1)
        return output

my_net = HastGCN(adj_sr.shape[0], adj_sr.shape[1], 9, 6, 2)
my_net.cuda()

#Train Model
criterion_mae_loss = nn.L1Loss()

optimizer = optim.Adam(params=my_net.parameters(), lr=0.0001, weight_decay=0.00)

# Train model
Epoch = 50
loss_min = 1e8
if os.path.exists('HastGCN_pems07.pkl'):
    my_net.load_state_dict(torch.load('HastGCN_pems07.pkl'))

my_net.train()
torch.autograd.set_detect_anomaly(True)
for epoch in range(Epoch):
    epoch_loss = 0.0
    start_time = time.time()
    for iter, (x, target) in enumerate(dataloader['train_loader'].get_iterator()):
        x = torch.Tensor(x).float().to(device) # x shape:[B, T_dim, N, D]
        target = torch.Tensor(target[:, -2:]).float().to(device) # y shape:[B, output_dim]
        my_net.zero_grad()
        predict_value = my_net(x, adj_rr, adj_sr, W, eind_ss, eind_rr)
        target = target.to(device)
        loss = criterion_mae_loss(predict_value, target)
        # print(loss)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print('time', np.mean(exe_time) / 1e3)
    end_time = time.time()

    print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss/(iter+1),
                                                                      (end_time - start_time) / 60))

    # Test Model
    # TODO: Visualize the Prediction Result
    # TODO: Measure the results with metrics MAE, MAPE, and RMSE
    my_net.eval()
    with torch.no_grad():
        total_rmse_loss = []
        total_mae_loss = []
        total_mape_loss = []
        for iter, (x, target) in enumerate(dataloader['val_loader'].get_iterator()):
            x = torch.Tensor(x).float().to(device) # x shape:[B, T_dim, N, D]
            target = torch.Tensor(target[:, -2:]).float().to(device) # y shape:[B, output_dim]
            # predict_value = my_net(x, W,device).to(torch.device("cpu"))  # [B, N, 1, D]
            predict_value = my_net(x,  adj_rr, adj_sr, W, eind_ss, eind_rr)
            predict_value, target = predict_value.detach().cpu().numpy(), target.cpu().numpy()

            mae_loss = np.abs(predict_value - target)
            rmse_loss = (predict_value - target) ** 2
            mape_loss = SMAPE_np(predict_value, target)

            total_rmse_loss.append(rmse_loss)
            total_mae_loss.append(mae_loss)
            total_mape_loss.append(mape_loss)

        if np.mean(total_mae_loss) < loss_min:
            loss_min = np.mean(total_mae_loss)
            torch.save(my_net.state_dict(), "HastGCN_pems07_1.pkl")
            print("RMSE: {:02.4f}".format(np.sqrt(np.mean(total_rmse_loss))))
            print("MAE: {:02.4f}".format(np.mean(total_mae_loss)))
            print("MAPE: {:02.4f}".format(np.mean(total_mape_loss)))


#Test model
my_net.eval()
my_net.load_state_dict(torch.load('HastGCN_pems07_1.pkl'))

criterion = nn.MSELoss()
criterion_mae_loss = nn.L1Loss()

with torch.no_grad():

    preds = []

    for iter, (x, target) in enumerate(dataloader['test_loader'].get_iterator()):
        x = torch.Tensor(x).float().to(device) # x shape:[B, T_dim, N, D]
        target = torch.Tensor(target).float().to(device) # y shape:[B, output_dim]

        # predict_value = my_net(x, W,device).to(torch.device("cpu"))  # [B, N, 1, D]
        predict_value = my_net(x, adj_rr, adj_sr, W, eind_ss, eind_rr)

        preds.append(predict_value.detach().cpu().numpy())

    preds = np.concatenate(preds)
    preds = preds[:dataloader['y_test'].shape[0]]
    print('MAE:', MAE_np(dataloader['y_test'][:, -2:], preds), 'MAPE:',
          SMAPE_np(dataloader['y_test'][:, -2:], preds),
          'RMSE:', RMSE_np(dataloader['y_test'][:, -2:], preds), 'R2:', R2_np(dataloader['y_test'][:, -2:], preds))

    print('MAE:', scaler[0].inverse_transform(MAE_np(dataloader['y_test'][:, -2], preds[:, -2])), 'MAPE:',
          SMAPE_np(dataloader['y_test'][:, -2], preds[:, -2]),
          'RMSE:', scaler[0].inverse_transform(RMSE_np(dataloader['y_test'][:, -2], preds[:, -2])), 'R2:',
          R2_np(dataloader['y_test'][:, -2], preds[:, -2]))
    print('MAE:', scaler[1].inverse_transform(MAE_np(dataloader['y_test'][:, -1], preds[:, -1])), 'MAPE:',
          SMAPE_np(dataloader['y_test'][:, -1], preds[:, -1]),
          'RMSE:', scaler[1].inverse_transform(RMSE_np(dataloader['y_test'][:, -1], preds[:, -1])), 'R2:',
          R2_np(dataloader['y_test'][:, -1], preds[:, -1]))
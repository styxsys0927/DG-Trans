import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

device = torch.device("cuda")

class GAT_layer(nn.Module):
    def __init__(self, N, T, d_model, dropout=0.1):
        super(GAT_layer, self).__init__()
        self.ws1 = nn.Parameter(torch.zeros(T))
        self.ws2 = nn.Parameter(torch.zeros(d_model, T))
        self.ws3 = nn.Parameter(torch.zeros(d_model))
        self.wd = nn.Parameter(torch.zeros(1, N))
        self.ws = nn.Parameter(torch.zeros(N, N))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, W, mask=False):
        Q = torch.einsum('t,btnd->bnd', self.ws1, src)
        K = torch.einsum('btnd,d->btn', src, self.ws3)
        att = torch.einsum('bqd,dt,btk->bqk', Q, self.ws2, K)
        att = F.relu(att + torch.einsum('s,mw->sw', torch.sum(W, dim=0), self.wd))
        if mask:
            att = att * W # mask sensor by roads
        att = F.softmax(self.ws * att, dim=0)
        src2 = src + self.dropout(torch.einsum('btnd,bna->btad', src, att))
        return src2

class TA_layer(nn.Module):
    def __init__(self, N, T, d_model, dropout=0.1):
        super(TA_layer, self).__init__()
        self.wt1 = nn.Parameter(torch.zeros(d_model))
        self.wt2 = nn.Parameter(torch.zeros(N, d_model))
        self.wt3 = nn.Parameter(torch.zeros(N))
        self.wt = nn.Parameter(torch.zeros(T, T))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        Q = torch.einsum('btnd,d->btn', src, self.wt1)
        K = torch.einsum('n,btnd->btd', self.wt3, src)
        att = F.softmax(torch.einsum('bqn,nd,bkd->bqk', Q, self.wt2, K), dim=0)
        src2 = src + self.dropout(torch.einsum('btnd,bta->band', src, att))
        return src2

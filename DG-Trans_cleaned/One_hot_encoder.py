# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:13:06 2020

@author: wb
"""
import torch
import torch.nn as nn


# 单位矩阵生成one-hot编码，线性层降维
class One_hot_encoder(nn.Module):
    def __init__(self, embed_size, time_num = 288):
        super(One_hot_encoder, self).__init__()
        
        self.time_num = time_num
        self.I = nn.Parameter(torch.eye(time_num, time_num, requires_grad=True))
        self.onehot_Linear = nn.Linear(time_num, embed_size)     # 线性层改变one hot编码维度

    def forward(self, i, N = 25, T = 12):
    
        if i%self.time_num+T > self.time_num :
            o1 = self.I[i%self.time_num : , : ]
            o2 = self.I[0 : (i+T)%self.time_num, : ]
            onehot = torch.cat((o1, o2), 0)
        else:        
            onehot = self.I[i%self.time_num: i%self.time_num+T, : ]
        
        #onehot = onehot.repeat(N, 1, 1)   
        onehot = onehot.expand(N, T, self.time_num)
        onehot = self.onehot_Linear(onehot)
        return onehot
    
class LSH_encoder(nn.Module):
    def __init__(self, n_buckets, B, T, D, n_hashes=1, dropout=0.1, n_epoch=100):
        # d_model%n_hashes should be 0 --> diff PE for diff dim
        super(LSH_encoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.n_buckets, self.n_hashes = n_buckets, n_hashes
        self.n_epoch = n_epoch
        self.hashT = torch.randn(B, T*D, self.n_hashes, n_buckets[0]//2).cuda()

    def get_pos(self, T, N, D, x, n_buckets):
        # self.hashT = torch.randn(1, T*D, self.n_hashes, n_buckets//2).cuda()

        rotated_vecs = torch.einsum('bf,bfhi->bhi', x, self.hashT)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        buckets = torch.argmax(rotated_vecs, dim=-1) # [B, H]
        tmp = []
        for i in range(T):
            tmp.append(((buckets+i)%n_buckets))
        tmp = torch.stack(tmp, dim=1)/n_buckets # [B, T, H]
        tmp = tmp[:, :, :].repeat(N, 1, D//self.n_hashes)

        return tmp

    def forward(self, x, iter):
        N, T, D = x.shape
        _x = torch.sum(x, dim=0).unsqueeze(0).reshape(1, -1)
        # if iter % self.n_epoch == 0:
        tmp = []
        for b in self.n_buckets:
            tmp.append(self.get_pos(T, N, D, _x, b))
        x = x.reshape(-1, T, D) + tmp[0]
        return self.dropout(x)
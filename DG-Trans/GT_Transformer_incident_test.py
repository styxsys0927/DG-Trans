# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020

@author: wb
"""

import torch
import torch.nn as nn
from GCN_models import GCN
from One_hot_encoder import One_hot_encoder, LSH_encoder
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        Nq, Tq, Cq = query.shape
        Nk, Tk, Ck = keys.shape

        values = values.reshape(Nk, Tk, self.heads, self.head_dim)
        keys   = keys.reshape(Nk, Tk, self.heads, self.head_dim)
        query  = query.reshape(Nq, Tq, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        energy = torch.einsum("qthd,kthd->qkth", [queries, keys])   # 空间self-attention
        if mask is not None:
            energy = energy * mask[:, :, None, None]
            energy[mask==0] = 1e6
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, N, T, heads)

        attention = torch.softmax((energy / (self.embed_size ** (1 / 2))).clamp(-5, 5), dim=1)  # 在K维做softmax，和为1
        # attention shape: (N, N, T, heads)

        out = torch.einsum("qkth,kthd->qthd", [attention, values]).reshape(
            Nq, Tq, self.heads * self.head_dim
        )
        # out after matrix multiply: (N, T, heads, head_dim)

        out = self.fc_out(out)
        # (N, T, embed_size)

        return out
    
class TSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N, Tq, C = query.shape
        Nk, Tk, Ck = keys.shape

        values = values.reshape(Nk, Tk, self.heads, self.head_dim)
        keys   = keys.reshape(Nk, Tk, self.heads, self.head_dim)
        query  = query.reshape(N, Tq, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])   # 时间self-attention
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, T, T, heads)
        if mask is not None:
            energy[mask[None, :, :, None].repeat(N, 1, 1, self.heads)==0] = -1e6
        attention = torch.softmax((energy / (self.embed_size ** (1 / 2))).clamp(-5, 5), dim=2)  # 在K维做softmax，和为1
        # attention shape: (N, query_len, key_len, heads)

        out = torch.einsum("nqkh,nkhd->nqhd", [attention, values]).reshape(
                N, Tq, self.heads * self.head_dim
        )
        # out after matrix multiply: (N, T, heads, head_dim)

        out = self.fc_out(out)
        # (N, T, embed_size)

        return out
    
    
class STransformer(nn.Module):
    def __init__(self, embed_size, t_len, heads, adj, sensor_dist, dataset, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.dataset = dataset
        self.adj_r = adj[1]
        self.adj_sr = adj[2]
        self.D_S = nn.Parameter(sensor_dist.unsqueeze(-1)) # act as position encoding
        self.D_E = nn.Parameter(torch.zeros_like(self.D_S)) # position encoding of event
        self.embed_liner = nn.Linear(1, embed_size)

        self.I = nn.Parameter(torch.Tensor(adj[1].shape[0], t_len, embed_size).float())
        nn.init.xavier_uniform_(self.I)
        
        self.attention = SSelfAttention(embed_size, heads)
        self.att_sr = SSelfAttention(embed_size, heads)
        self.att_rr = SSelfAttention(embed_size, heads)
        self.att_rs = SSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, d, adj_rs=None, mask=True):
                
        # Spatial Embedding 部分
        Nq, T, C = query.shape
        Nk = key.shape[0]
        D_S = self.embed_liner(self.D_S)
        D_S = D_S.expand(T, Nk, C)
        D_S = D_S.permute(1, 0, 2)

        # Spatial Transformer 部分
        query, key = query+D_S if Nq==Nk else query, key+D_S
        if mask:
            H = self.att_sr(value, key, self.I, mask=self.adj_sr.T)
            H = self.att_rr(H, H, H, mask=self.adj_r)
            attention = self.att_rs(H, H, query, mask=None if adj_rs is None else adj_rs.T)
        else:
            H = self.att_sr(value, key, self.I)
            H = self.att_rr(H, H, H)
            attention = self.att_rs(H, H, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        out = self.fs(U_S)

        return out

class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, t_len, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
        self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding
        # self.lsh = LSH_encoder([12*24*7], 1, t_len, embed_size, n_hashes=1)#embed_size)
        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, tq, tk=-1, mask=False):
        N, Tq, C = query.shape
        Tk = key.shape[1]
        tk = tq if tk<0 else tk
        tq_inds = ((self.time_num+tq)+torch.arange(0, Tq).to(device))%self.time_num
        tk_inds = ((self.time_num+tk)+torch.arange(0, Tk).to(device))%self.time_num
        D_Tq = self.temporal_embedding(tq_inds)    # temporal embedding选用nn.Embedding
        D_Tq = D_Tq.expand(N, Tq, C)
        D_Tk = self.temporal_embedding(tk_inds)    # temporal embedding选用nn.Embedding
        D_Tk = D_Tk.expand(N, Tk, C)

        # temporal embedding加到query。 原论文采用concatenated
        query, key = query + D_Tq, key + D_Tk

        t_mask = torch.tril(torch.ones(Tq, Tk)).to(device) if mask else None
        attention = self.attention(value, key, query, t_mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, sensor_dist, t_len, time_num, dataset, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, t_len, heads, adj, sensor_dist, dataset, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, d, t):
        # Add skip connection,run through normalization and finally dropout
        x = self.STransformer(value, key, query, d, mask=True)
        x1 = self.norm1(x + query)
        x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1, t) + x1) )

        return x2

class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(self, embed_size, num_layers, heads, adj, sensor_dist, t_len, time_num, device, forward_expansion, dropout, dataset):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList([STTransformerBlock(embed_size, heads, adj, sensor_dist, t_len, time_num, dataset, dropout=dropout,
                    forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, d, t):
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out, d, t)
        return out


class Decoder(nn.Module):
    def __init__(self, embed_size, heads, adj, sendor_dist, t_len, time_num, dataset, dropout, forward_expansion):
        super(Decoder, self).__init__()
        self.n_road, self.n_node = adj[2].shape[1], adj[2].shape[0]
        self.T_Event = nn.Parameter(torch.zeros(self.n_node, 1, embed_size).to(device))
        self.S_Event = nn.Parameter(torch.zeros(1, 1, embed_size).to(device))
        self.STransformer = STransformer(embed_size, 1, heads, adj, sendor_dist, dataset, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, r, d, t):
        # query is the event
        adj_rs = torch.zeros(self.n_road, 1).to(device)
        adj_rs[r, :] = 1
        x = self.norm1(self.TTransformer(value, key, self.T_Event, t, t-3) + self.T_Event)
        x1 = self.dropout(self.norm2(self.STransformer(x, x, self.S_Event, d, adj_rs, mask=True) + self.S_Event))

        return x1

class Transformer(nn.Module):
    def __init__(self, adj, sensor_dist, dataset, embed_size=64, num_layers=3, heads=2, time_num=288, forward_expansion=4, dropout=0, device="cpu"):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embed_size, num_layers, heads, adj, sensor_dist, 6, time_num, device, forward_expansion, dropout, dataset)
        self.decoder1 = TTransformer(embed_size, heads, 6, time_num, dropout, forward_expansion)
        self.decoder2 = TTransformer(embed_size, heads, 6, time_num, dropout, forward_expansion)
        self.decoder = Decoder(embed_size, heads, adj, sensor_dist, 6, time_num, dataset, dropout=dropout, forward_expansion=forward_expansion)
        self.device = device

        self.norm = nn.LayerNorm(embed_size)

    def forward(self, src, r, d, t):
        x = self.encoder(src, d, t-3)
        x = self.norm(self.decoder1(x, x, x, t-3, mask=True) + x)
        x = self.norm(self.decoder2(x, x, x, t-3, mask=True) + x)
        x = self.decoder(x, x, r, d, t)
        return x, 0, 0

class Pooling(nn.Module):
    def __init__(self, T, in_c, heads, adj, dataset, dropout=0, forward_expansion=4, hidden_c=256, hidden_c_2=128, out_c=2):
        super(Pooling, self).__init__()
        self.in_c = in_c
        self.c_class = nn.Sequential(
            nn.Linear(in_c, 128),
            nn.Sigmoid(),
            nn.Linear(128, 13),
            nn.Softmax(dim=-1))
        self.linear_1 = nn.Linear(in_c, hidden_c)
        self.linear_2 = nn.Linear(hidden_c, hidden_c_2)
        self.linear_3 = nn.Linear(hidden_c_2, out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        c_class = self.c_class(x)
        output_2 = self.act(self.linear_1(x))
        output_2 = self.act(self.linear_2(output_2))
        output_3 = self.linear_3(output_2)
        return output_3.squeeze(0), c_class.squeeze(0)

class STTransformer(nn.Module):
    def __init__(
        self, 
        adj,
        sensor_dist,
        dataset,
        in_channels = 2,
        embed_size = 64, 
        time_num = 288,
        num_layers = 3,
        T_dim = 12,
        T_dim_af = 6,
        heads = 2,
    ):        
        super(STTransformer, self).__init__()
        self.n_road, self.n_node = adj[2].shape[1], adj[2].shape[0]
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, (1, 1))
        self.Transformer = Transformer(
            adj,
            sensor_dist,
            dataset,
            embed_size, 
            num_layers, 
            heads, 
            time_num,
            device=device
        )

        self.pooling = Pooling(T_dim, embed_size, heads, adj, dataset)

    
    def forward(self, x, r, d, t):
        # input x shape[ C, N, T] 
        # C:通道数量。  N:传感器数量。  T:时间数量

        input_Transformer = self.conv1(x)        
        input_Transformer = input_Transformer.squeeze(0)
        input_Transformer = input_Transformer.permute(1, 2, 0)
        
        #input_Transformer shape[N, T, C]
        output_Transformer, x1, x2 = self.Transformer(input_Transformer, r, d, t)
        output_Transformer = output_Transformer.permute(1, 0, 2)
        #output_Transformer shape[T, N, C]

        out, c = self.pooling(output_Transformer)
        return out, torch.zeros(1, self.n_road), c, (0, 0)
        # return out shape: [N, output_dim]
    


    

    
    
    
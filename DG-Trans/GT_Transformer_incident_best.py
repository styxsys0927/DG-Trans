# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020

@author: wb
"""

import torch
import torch.nn as nn
from GCN_models import GCN
from One_hot_encoder import One_hot_encoder, LSH_encoder
from config import dataset

global exe_time
exe_time=[]

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

        # Split the embedding into self.heads different pieces
        values = values.reshape(Nk, Tk, self.heads, self.head_dim)  #embed_size维拆成 heads×head_dim
        keys   = keys.reshape(Nk, Tk, self.heads, self.head_dim)
        query  = query.reshape(Nq, Tq, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        energy = torch.einsum("qthd,kthd->qkth", [queries, keys])   # 空间self-attention
        if mask is not None:
            energy = energy * mask[:, :, None, None]
            energy[mask==0] = -1e6
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, N, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax((energy / (self.embed_size ** (1 / 2))).clamp(-5, 5), dim=1)  # 在K维做softmax，和为1
        # attention shape: (N, N, T, heads)

        out = torch.einsum("qkth,kthd->qthd", [attention, values]).reshape(
            Nq, Tq, self.heads * self.head_dim
        )        
        # attention shape: (N, N, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        if (mask is not None and torch.isnan(mask).sum()) or torch.isnan(energy).sum() or torch.isnan(attention).sum() or torch.isnan(out).sum():
            print(torch.isnan(mask).sum(),torch.isnan(energy).sum(),torch.isnan(attention).sum(),torch.isnan(out).sum())
            exit(111)
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
        N, T, C = query.shape
        Nk, Tk, Ck = keys.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(Nk, Tk, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        keys   = keys.reshape(Nk, Tk, self.heads, self.head_dim)
        query  = query.reshape(N, T, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])   # 时间self-attention
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, T, T, heads)


        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        if mask is not None:
            energy[mask[None, :, :, None].repeat(N, 1, 1, self.heads)==0] = -1e6
        attention = torch.softmax((energy / (self.embed_size ** (1 / 2))).clamp(-5, 5), dim=2)  # 在K维做softmax，和为1
        # attention shape: (N, query_len, key_len, heads)

        out = torch.einsum("nqkh,nkhd->nqhd", [attention, values]).reshape(
                N, T, self.heads * self.head_dim
        )
        # attention shape: (N, T, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        return out
    
    
class STransformer(nn.Module):
    def __init__(self, embed_size, time_num, heads, adj, dataset, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.dataset = dataset
        self.adj = adj[0]
        self.adj_r = adj[1]
        self.adj_sr = adj[2]
        self.D_S = nn.Parameter(adj[2]) # act as position encoding
        self.embed_liner = nn.Linear(adj[2].shape[1], embed_size)

        self.I = nn.Parameter(torch.Tensor(adj[1].shape[0], time_num, embed_size).float())
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

    def forward(self, value, key, query, mask=True):
        # Spatial Embedding 部分
        N, T, C = query.shape
        D_S = self.embed_liner(self.D_S)
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)

        # Spatial Transformer 部分
        query, key, value = query+D_S, key+D_S, value+D_S
        # attention = self.attention(value, key, query)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        if mask:
            H = self.att_sr(value, key, self.I, mask=self.adj_sr.T)
            H = self.att_rr(H, H, H, mask=self.adj_r)
        else:
            H = self.att_sr(value, key, self.I)
            H = self.att_rr(H, H, H)

        attention = self.att_rs(H, H, query)

        # end.record()
        # # Waits for everything to finish running
        # torch.cuda.synchronize()
        # exe_time.append(start.elapsed_time(end))  # milliseconds

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
        self.lsh = LSH_encoder([12*24*7], 1, t_len, embed_size, n_hashes=1)#embed_size)

        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t, mask=False):
        N, Tq, C = query.shape
        N, Tk, C = key.shape
        
        # D_T = self.one_hot(t, N, T).to(device)                          # temporal embedding选用one-hot方式 或者
        # D_T = self.temporal_embedding(torch.arange(0, T).to(device))    # temporal embedding选用nn.Embedding
        # D_T = D_T.expand(N, T, C)
        D_T = self.lsh(query, t)

        # temporal embedding加到query。 原论文采用concatenated
        # query = query + D_T
        # if key.shape[1] == T:
        #     key, value = key+D_T, value+D_T

        t_mask = torch.tril(torch.ones(Tq, Tk)).to(device) if mask else None
        attention = self.attention(value, key, query, t_mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Decoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(self, embed_size, num_layers, heads, t_len, time_num, device, forward_expansion, dropout, dataset):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.TT_self1 = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        # self.TT_self2 = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.TT_mu = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        x = self.TT_self1(query, query, query, t, mask=True)

        # x = self.TT_self2(x, x, x, t, mask=True)
        x1 = self.norm1(x + query)
        x2 = self.dropout(self.norm2(self.TT_mu(value, key, x1, t) + x1))
        return x2

class Transformer2(nn.Module):
    def __init__(self, adj, dataset, embed_size=64, num_layers=3, heads=2, time_num=288, forward_expansion=4, dropout=0, device="cpu"):
        super(Transformer2, self).__init__()
        self.STransformer = STransformer(embed_size, time_num, heads, adj, dataset, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size*2, heads, 12, time_num, dropout, forward_expansion)
        self.TTransformer_full = TTransformer(embed_size*2, heads, 18, time_num, dropout, forward_expansion)
        self.decoder1 = Decoder(embed_size*2, num_layers, heads, 6, time_num, device, forward_expansion, dropout, dataset)
        self.decoder2 = Decoder(embed_size*2, num_layers, heads, 6, time_num, device, forward_expansion, dropout, dataset)
        self.device = device

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size*2)

        self.fcn = nn.Sequential(nn.Linear(2 * embed_size, embed_size), nn.Sigmoid())
        self.ln = nn.Linear(embed_size, 2) # input feature size is 2

    def sencode(self, value, key, query, mask=True):
        x = self.STransformer(value, key, query, mask=mask)
        x1 = self.norm1(x + query)
        return x1

    def forward(self, src, t, mask=True):
        senc_src = self.sencode(src, src, src, mask=mask)
        src, tgt = senc_src, senc_src[:, 12:, :]
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        src1 = torch.cat((src, c), dim=2)
        x = self.norm2(self.TTransformer_full(src1, src1, src1, t) + src1)

        tgt1 = tgt.repeat(1, 1, 2)
        x1 = self.fcn(self.decoder1(x, x, tgt1, t))
        # Phase 2 - With anomaly scores
        c = (senc_src - x1.repeat(1, 3, 1)).abs()

        src2 = torch.cat((senc_src, c), dim=2)
        x = self.norm2(self.TTransformer_full(src2, src2, src2, t) + src2)
        # x = torch.cat((x, c), dim=2)
        x2 = self.fcn(self.decoder2(x, x, tgt1, t))
        return x, c, self.ln(x1), self.ln(x2)

class Pooling(nn.Module):
    def __init__(self, T, in_c, hidden_c=256, hidden_c_2=128, out_c=2):
        super(Pooling, self).__init__()

        self.conv_T = nn.Conv2d(T, 1, (1, 1))
        self.linear_1 = nn.Linear(in_c, hidden_c)
        self.linear_2 = nn.Linear(hidden_c, hidden_c_2)
        self.linear_3 = nn.Linear(hidden_c_2, out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        x = self.act(self.conv_T(x)).squeeze(1) # (1, N, C)
        output_1 = torch.sum(x, dim=1).squeeze(1) # (1, C)
        # 降到batch size*2维。
        output_3 = self.act(self.linear_1(output_1))
        output_3 = self.act(self.linear_2(output_3))
        output_3 = self.linear_3(output_3)
        return output_3

class STTransformer(nn.Module):
    def __init__(
        self, 
        adj,
        dataset,
        in_channels = 2,
        embed_size = 64, 
        time_num = 288,
        num_layers = 3,
        T_dim = 12,
        output_T_dim = 3,  
        heads = 2,
    ):        
        super(STTransformer, self).__init__()
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, (1, 1))
        self.Transformer = Transformer2(
            adj,
            dataset,
            embed_size, 
            num_layers, 
            heads, 
            time_num,
            device=device
        )

        self.pooling = Pooling(output_T_dim, embed_size*2)

    
    def forward(self, x, t):
        # input x shape[ C, N, T] 
        # C:通道数量。  N:传感器数量。  T:时间数量

        input_Transformer = self.conv1(x)        
        input_Transformer = input_Transformer.squeeze(0)
        input_Transformer = input_Transformer.permute(1, 2, 0)
        
        #input_Transformer shape[N, T, C]
        output_Transformer, c, x1, x2 = self.Transformer(input_Transformer, t)
        # residual = torch.cat((torch.zeros(residual.shape[0], 12, residual.shape[2]).to(device), residual), dim=1)
        # output_Transformer = torch.cat((output_Transformer, residual), dim=-1)
        output_Transformer = output_Transformer.permute(1, 0, 2)
        #output_Transformer shape[T, N, C*2]

        out = self.pooling(output_Transformer)
        return out, c, (x1, x2)
        # return out shape: [N, output_dim]
    


    

    
    
    
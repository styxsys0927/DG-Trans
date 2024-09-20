# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020

@author: wb
"""

import torch
import torch.nn as nn
from GCN_models import GCN
from One_hot_encoder import One_hot_encoder, LSH_encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    def forward(self, values, keys, query):
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
    
    
class GraphSFWT(nn.Module):
    def __init__(self, embed_size, adj, hidden_c=1024, hidden_c_2=512):
        super(GraphSFWT, self).__init__()
        self.adj = adj
        self.linear_1 = nn.Linear(embed_size, hidden_c)
        self.linear_2 = nn.Linear(hidden_c, hidden_c_2)
        self.linear_3 = nn.Linear(hidden_c_2, embed_size)
        self.act = nn.ReLU()

    def forward(self, data):
        output_1 = self.act(torch.einsum("nm,mtd->ntd", [self.adj, data]))  # [N, N], [N, T, D]
        output_2 = self.act(torch.einsum("nm,mtd->ntd", [self.adj, output_1]))

        output_3 = self.act(self.linear_1(output_2))
        output_3 = self.act(self.linear_2(output_3))
        output_3 = self.linear_3(output_3)

        return output_3

class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, t_len, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, t_len, time_num, dataset, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = GraphSFWT(embed_size, adj)
        self.TTransformer = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, t, mask=True):
        # Add skip connection,run through normalization and finally dropout
        x = self.STransformer(query)
        x1 = self.norm1(x + query)
        x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1, t) + x1) )

        return x2

class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(self, embed_size, num_layers, heads, adj, t_len, time_num, device, forward_expansion, dropout, dataset):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList([STTransformerBlock(embed_size, heads, adj, t_len, time_num, dataset, dropout=dropout,
                    forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out, t, mask=True)
        # out = self.layers[-1](out, out, out, t, mask=False)
        return out

class Decoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(self, embed_size, num_layers, heads, t_len, time_num, device, forward_expansion, dropout, dataset):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.TT_self = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.TT_mu = TTransformer(embed_size, heads, t_len, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        x = self.TT_self(query, query, query, t)
        x1 = self.norm1(x + query)
        x2 = self.dropout(self.norm2(self.TT_mu(value, key, x1, t) + x1))
        return x2

class Transformer2(nn.Module):
    def __init__(self, adj, dataset, embed_size=64, num_layers=3, heads=2, time_num=288, forward_expansion=4, dropout=0, device="cpu"):
        super(Transformer2, self).__init__()
        self.STransformer = GraphSFWT(embed_size, adj)
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
        x = self.STransformer(query)
        x1 = self.norm1(x + query)
        return x1

    def forward(self, src, t, mask=True):
        senc_src = self.sencode(src, src, src, mask=mask)
        src, tgt = senc_src, senc_src[:, 6:, :]
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        src1 = torch.cat((src, c), dim=2)
        x = self.norm2(self.TTransformer_full(src1, src1, src1, t) + src1)

        tgt1 = tgt.repeat(1, 1, 2)
        x1 = self.fcn(self.decoder1(x, x, tgt1, t))
        # Phase 2 - With anomaly scores
        c = x1.repeat(1, 3, 1) - senc_src #torch.cat((torch.zeros_like(src), x1 - tgt), dim=1)
        src2 = torch.cat((senc_src, c), dim=2)
        x = self.norm2(self.TTransformer_full(src2, src2, src2, t) + src2)
        # x = torch.cat((x, c), dim=2)
        x2 = self.fcn(self.decoder2(x, x, tgt1, t))
        return x, self.ln(x1), self.ln(x2)

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
        output_Transformer, x1, x2 = self.Transformer(input_Transformer, t)
        # residual = torch.cat((torch.zeros(residual.shape[0], 12, residual.shape[2]).to(device), residual), dim=1)
        # output_Transformer = torch.cat((output_Transformer, residual), dim=-1)
        output_Transformer = output_Transformer.permute(1, 0, 2)
        #output_Transformer shape[T, N, C*2]

        out = self.pooling(output_Transformer)
        return out, (x1, x2)
        # return out shape: [N, output_dim]
    


    

    
    
    
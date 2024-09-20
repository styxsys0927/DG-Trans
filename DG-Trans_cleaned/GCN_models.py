import torch.nn as nn
import torch.nn.functional as F
from layers.layers import SimpleGraphConvolution, MLPEncoder
import time


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = SimpleGraphConvolution(nfeat, nhid)
        self.gc2 = SimpleGraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # N_object*Nhid
        x = self.gc2(x, adj)
        return x#F.log_softmax(x, dim=1)


class GCN_evolution(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_evolution, self).__init__()

        self.gc1 = MLPEncoder(nfeat, nhid, nhid)
        self.gc2 = MLPEncoder(nhid, nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj, adj.T)) # 1*N_object*Nhid
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, adj.T).squeeze() # N_object*Nhid
        return F.log_softmax(x, dim=1)

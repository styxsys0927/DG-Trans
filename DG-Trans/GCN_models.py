import torch.nn as nn
import torch.nn.functional as F
from layers import SimpleGraphConvolution


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

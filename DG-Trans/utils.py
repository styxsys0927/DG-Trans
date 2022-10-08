import torch
import numpy as np

from scipy import signal
from scipy.sparse import csgraph
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#AGWN wavelet kernel generation
def sfwt_wavelet(A, tau=6):
    # '''
    # A : adjacency matrix
    # tau: scale factor
    # h: continuous wavelet filter
    # '''

    # Construct graph wavelet filter, W
    h = signal.ricker(1000, 100)[500:]
    W = np.zeros(A.shape).astype(float)
    # Generate geodesic distances
    A = A.cpu().numpy()
    spath = csgraph.shortest_path(A, directed=False, unweighted=True).astype(int)
    spath[spath<0] = A.shape[0]+1 # longest SP is the number of nodes
    # Resample filter
    if not (h.size % tau):
        hm = h.reshape(tau, -1).mean(axis=1)
    else:
        hm = h[:-(h.size % tau)].reshape(tau, -1).mean(axis=1)

    for i in range(W.shape[0]):
        # N: histogram of distances from i
        # N_t: Number of vertices within k hops of i for all k < tau
        N = np.bincount(spath[i, :])
        N_t = np.where(spath[i, :] < tau, N[spath[i, :]], i)
        mask = (spath[i, :] < tau)
        # a : wavelet coefficients
        a = np.zeros_like(spath[i, :]).astype(float)
        a[mask] = hm[spath[i, :][mask]] / N_t[mask].astype(float)
        # W[:, i] = a+0.001
        W[:, i] = a

    return W


#generate weight for the adjacent matrix
def process_graph(graph_data):
    N = graph_data.size(0)
    matrix_i = torch.eye(N)
    matrix_i = matrix_i.to(torch.device("cuda"))

    graph_data += matrix_i  # A~ [N, N]

    degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
    degree_matrix = degree_matrix.pow(-1)
    degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

    degree_matrix = torch.diag(degree_matrix)  # [N, N]

    return torch.mm(degree_matrix, graph_data)

# Common practise for initialization.
def weights_init(model):

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
            nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)

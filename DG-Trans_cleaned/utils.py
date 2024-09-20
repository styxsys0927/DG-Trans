import torch
import numpy as np

from scipy import signal
from scipy.sparse import csgraph
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

config = {
    'smoothness_ratio': 0.4, # 0.4, IGL: 0.4
    'degree_ratio': 0, # 0
    'sparsity_ratio': 0.2, # 0.2, IGL: 0.2
    'graph_learn_ratio': 0 # 0
}

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
def process_graph(graph_data, device='cuda'):
    N = graph_data.size(0)
    # matrix_i = torch.eye(N)
    # matrix_i = matrix_i.to(torch.device(device))
    #
    # graph_data += matrix_i  # A~ [N, N]

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

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def SMAPE_torch(y,out):
    return ((out-y).abs()/(out.abs()+y.abs())*2).mean()

def SMAPE_np(y, out):
    return np.mean(np.abs(y-out)/(np.abs(y)+np.abs(out))*2)

def R2_np(y, out):
    return 1-np.sum((y-out)**2)/np.sum((y-np.mean(y))**2)

def MAE_np(y, out):
    return np.mean(np.abs(y-out))

def RMSE_np(y, out):
    return np.sqrt(np.mean((y-out)**2))


from torch.distributions import MultivariateNormal as MVN
from torch.nn.modules.loss import _Loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss_md(pred, target, noise_var)


def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1]).cuda()
    logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())  # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss

def wmae_loss(out, y, bias=0.001):
    l0 = (torch.exp(y[:,0])+bias)*torch.abs(y[:,0] - out[:,0])#torch.abs(y[:,0] - out[:,0])
    l1 = (torch.exp(y[:,1])+bias)*torch.abs(y[:,1] - out[:,1])
    return torch.mean(l0+l1)

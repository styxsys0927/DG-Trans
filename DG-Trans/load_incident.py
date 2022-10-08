import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.utils.data import Dataset#, DataLoader
import numpy as np
import pandas as pd

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def normalize_digraph(A):
    Dl = np.sum(A, 0)  #计算邻接矩阵的度
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1) #由每个点的度组成的对角矩阵
    AD = np.dot(A, Dn)
    return AD

def load_incident(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    # Load raw data
    gt_file = np.load('../graphtransformer-main/data/'+dataset_dir+'/graph_label_full.npy') # 'eid', '5-min', 'rid', 'type_id', 'dists', 'dur (s)', 'q_len (m)'
    data_file = np.load('../graphtransformer-main/data/'+dataset_dir+'/graph_data_full.npy')
    sensor_dist =  pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/sensors_dists.csv', header=0, index_col=0)['dists'].to_numpy()
    filter = (gt_file[:,-1]<60000)&(gt_file[:,-2]<12000)
    gt_file, data_file = gt_file[filter], data_file[filter]

    data['A'] = torch.from_numpy(np.load('../graphtransformer-main/data/'+dataset_dir+'/adj_ss.npy'))
    data['A'] = data['A'] + torch.eye(data['A'].shape[0])

    data['A_rr'] = pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/adj_rr.csv', index_col=0, header=0).to_numpy()
    data['A_sr'] = pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/adj_sr.csv', index_col=0, header=0).to_numpy()
    data['A_rr'] = data['A_rr'] + np.eye(data['A_rr'].shape[0])

    road_idx = np.load('../graphtransformer-main/data/'+dataset_dir+'/road_idx.npy')

    gt=gt_file[:, 1:] # exclude event id
    gt_m0, gt_m1 = gt[:, -2].max(), gt[:, -1].max()
    for i in range(gt.shape[0]):
        r = gt[i, 1].astype(int)
        gt[i, 1] = np.where(road_idx==r)[0][0]
    gt[:, -2], gt[:, -1] = gt[:, -2] / gt_m0, gt[:, -1] / gt_m1

    # rescale dists to [0, 1]
    data['sensor_dist'] = np.zeros_like(data['A_sr'])
    for r in range(len(road_idx)):
        r_event, r_sensor = gt[:, 1]==r, data['A_sr'][:, r]>0
        m_dist = np.concatenate([gt[:, 3][r_event], sensor_dist[r_sensor]]).max()
        gt[:, 3][r_event], data['sensor_dist'][r_sensor, r] = gt[:, 3][r_event]/m_dist, sensor_dist[r_sensor]/m_dist*r # +r to distinguish roads

    #generate train,valid,test data
    num_tr, num_te = int(data_file.shape[0]*0.7), int(data_file.shape[0]*0.9)
    print(data_file.shape[0], num_tr, num_te, 'cases, label scale', gt_m0, gt_m1)
    data['x_train'], data['y_train'] = data_file[:num_tr], gt[:num_tr]
    data['x_test'], data['y_test'] = data_file[num_tr:num_te], gt[num_tr:num_te]
    data['x_val'], data['y_val'] = data_file[num_te:], gt[num_te:]
    scalers = [gt_m0, gt_m1]
    for i in range(data['x_train'].shape[-1]):
        scalers.append(StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std()))
    # Data format
    for category in ['train', 'val', 'test']:
        for i in range(data['x_train'].shape[-1]):
            data['x_' + category][..., i] = scalers[i+2].transform(data['x_' + category][..., i])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalers
    return data

def load_incident2(dataset_dir):
    # process data for lasso and SVR
    data = {}
    # Load raw data
    gt_file = np.load('../graphtransformer-main/data/'+dataset_dir+'/graph_label_full.npy') # 'eid', '5-min', 'rid', 'type_id', 'dists', 'dur (s)', 'q_len (m)'
    data_file = np.load('../graphtransformer-main/data/'+dataset_dir+'/graph_data_full.npy')
    sensor_dist =  pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/sensors_dists.csv', header=0, index_col=0)['dists'].to_numpy()
    # filter = (gt_file[:,-1]<60000)&(gt_file[:,-2]<12000)
    # gt_file, data_file = gt_file[filter], data_file[filter]

    data['A'] = torch.from_numpy(np.load('../graphtransformer-main/data/'+dataset_dir+'/adj_ss.npy'))
    data['A'] = data['A'] + torch.eye(data['A'].shape[0])

    data['A_rr'] = pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/adj_rr.csv', index_col=0, header=0).to_numpy()
    data['A_sr'] = pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/adj_sr.csv', index_col=0, header=0).to_numpy()
    data['A_rr'] = data['A_rr'] + np.eye(data['A_rr'].shape[0])

    road_idx = np.load('../graphtransformer-main/data/'+dataset_dir+'/road_idx.npy')

    data['data'] = []
    gt=gt_file[:, 1:] # exclude event id
    gt_m0, gt_m1 = gt[:, -2].max(), gt[:, -1].max()
    for i in range(gt.shape[0]):
        r = gt[i, 1].astype(int)
        gt[i, 1] = np.where(road_idx==r)[0][0]
        tmp = sensor_dist.copy()
        tmp[(data['A_sr'][:, int(gt[i, 1])]<1e-6)|(gt[i, 3]<sensor_dist)] = np.inf
        sid = tmp.argmin()
        data['data'].append(data_file[i, 12, sid, :].reshape(-1))
        # data['data'].append(data_file[i, :, :, :].reshape(-1, 2).mean(axis=0))
    data['data'] = np.array(data['data'])
    gt[:, -2], gt[:, -1] = gt[:, -2] / gt_m0, gt[:, -1] / gt_m1

    #generate train,valid,test data
    num_tr, num_te = int(data_file.shape[0]*0.7), int(data_file.shape[0]*0.9)
    print(data_file.shape[0], num_tr, num_te, 'cases, label scale', gt_m0, gt_m1)
    data['x_train'], data['y_train'] = data['data'][:num_tr], gt[:num_tr, -2:]
    data['x_test'], data['y_test'] = data['data'][num_tr:num_te], gt[num_tr:num_te, -2:]
    data['x_val'], data['y_val'] = data['data'][num_te:], gt[num_te:, -2:]
    scalers = [gt_m0, gt_m1]
    for i in range(data['x_train'].shape[-1]):
        scalers.append(StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std()))
    # Data format
    for category in ['train', 'val', 'test']:
        for i in range(data['x_train'].shape[-1]):
            data['x_' + category][..., i] = scalers[i+2].transform(data['x_' + category][..., i])
    data['scaler'] = scalers
    return data
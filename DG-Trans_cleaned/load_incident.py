import pickle
import numpy as np
import os
import scipy.sparse as sp
# import torch
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, cs=None):
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
        self.cs = None
        if cs is not None:
            c_padding = np.repeat(cs[-1:], num_padding, axis=0)
            cs = np.concatenate([cs, c_padding], axis=0)
            self.cs = cs

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        if self.cs is not None:
            self.cs = self.cs[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                if self.cs is not None:
                    c_i = self.cs[start_ind: end_ind, ...]
                    yield (x_i, y_i, c_i)
                else:
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

class MinMaxScaler():
    """
    minmax the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max-self.min)

    def inverse_transform(self, data):
        return data * (self.max-self.min) + self.min

class LogMinMaxScaler():
    """
    minmax the input
    """

    def __init__(self, min, max):
        self.min = np.log(min)
        self.max = np.log(max)

    def transform(self, data):
        return (np.log(data) - self.min) / (self.max-self.min)

    def inverse_transform(self, data):
        return np.exp(data * (self.max-self.min) + self.min)

def normalize_digraph(A):
    Dl = np.sum(A, 0)  #计算邻接矩阵的度
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1) #由每个点的度组成的对角矩阵
    AD = np.dot(A, Dn)
    return AD

def load_incident(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, k=2):
    data = {}
    # Load raw data
    gt_file = np.load('../graphtransformer-main/data/'+dataset_dir+'/graph_label_full_v3.npy') # 'eid', '5-min', 'rid', 'type_id', 'dists', 'dur (s)', 'q_len (m)'
    data_file = np.load('../graphtransformer-main/data/'+dataset_dir+'/graph_data_full.npy')
    # filter = (gt_file[:,-2]>1800) # 30 * 60 seconds
    gt, data_file = gt_file, data_file
    data_file = data_file[:, 6:15, :, :] # 30 mins before the incident and 15 minutes after the incident

    data['A_rr'] = pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/adj_rr.csv', index_col=0, header=0).to_numpy()
    data['A_sr'] = pd.read_csv('../graphtransformer-main/data/'+dataset_dir+'/adj_sr.csv', index_col=0, header=0).to_numpy()
    data['A_rr'] = data['A_rr'] + np.eye(data['A_rr'].shape[0])

    scalers = []
    num_tr, num_te = int(data_file.shape[0]*0.7), int(data_file.shape[0]*0.9)
    print(f"Total number of samples: {data_file.shape[0]}. {num_tr} are used for training and {num_te-num_tr} for testing.", data_file.shape, gt.shape)
    scalers.append(MinMaxScaler(min=gt[:num_tr, -2].min(), max=gt[:num_tr, -2].max()))
    scalers.append(MinMaxScaler(min=gt[:num_tr, -1].min(), max=gt[:num_tr, -1].max()))
    gt[:, -2], gt[:, -1] = scalers[0].transform(gt[:, -2]), scalers[1].transform(gt[:, -1])

    #generate train,valid,test data
    data['x_train'], data['y_train'] = data_file[:num_tr], gt[:num_tr]
    data['x_test'], data['y_test'] = data_file[num_tr:num_te], gt[num_tr:num_te]
    data['x_val'], data['y_val'] = data_file[num_te:], gt[num_te:]
    for i in range(data['x_train'].shape[-1]):
        scalers.append(StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std()))

    for category in ['train', 'val', 'test']:
        for i in range(data['x_train'].shape[-1]):
            data['x_' + category][..., i] = scalers[i+2].transform(data['x_' + category][..., i])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scalers

    data['train_loader'].shuffle()
    data['val_loader'].shuffle()
    return data


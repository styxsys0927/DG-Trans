import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib import rcParams

class EVTScaler():
    """
    extreme value theory scaler (KDD 22)
    constraints: scale>0, 1+shape*y/scale>0
    """

    def __init__(self, shape, scale, threshold=0):
        self.threshold, self.shape, self.scale = threshold, shape, scale

    def transform(self, y):
        if self.shape == 0:
            return 1/self.scale*np.exp(-y/self.scale)
        return 1/self.scale*(1+self.shape*y/self.scale)^(-1/self.shape-1)

    def inverse_transform(self, y):
        if self.shape == 0:
            return 1/self.scale*np.exp(-y/self.scale)
        return 1/self.scale*(1+self.shape*y/self.scale)^(-1/self.shape-1)

def plot_hist(ax, label, data, xlabel=None, ylabel=None):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-0, 0))
    ax.yaxis.set_major_formatter(formatter)

    sns.histplot(data, ax=ax, bins=10, kde=True, line_kws={'lw': 5}, label=label)
    ax.lines[0].set_color('darkorange')
    vals = np.log(np.array([rec.get_height() for rec in ax.patches]))
    vals[vals < 0] = 0
    norm = plt.Normalize(vals.min(), vals.max())
    colors = plt.cm.Blues(norm(vals))
    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_yticklabels(ax.get_yticks(), size=20)
    # ax.set_xticklabels(ax.get_xticks(), size=20)
    ax.legend(fontsize=20)

def plot_scatter(label, data, hue, xlabel=None, ylabel=None):
    if data.shape[0] <= 0:
        return
    sns.set(style='ticks')

    fg = sns.FacetGrid(data=data, hue=hue, aspect=1.0, cmap = "Spectral")
    fg.map(plt.scatter, 'Dur-'+label, 'Len-'+label).add_legend()

def plot_scatter_continuous(label, data, hue, xlabel=None, ylabel=None):
    cmap = sns.color_palette("Spectral", as_cmap=True)#sns.cubehelix_palette(rot=-.2, as_cmap=True)
    g = sns.relplot(
        data=data,
        x='Dur-'+label, y='Len-'+label,
        hue="dists",
        palette=cmap)
    leg = g._legend
    for t in leg.texts:
        # truncate label text to 4 characters
        p = t.get_text().find('.')
        if p != -1:
            t.set_text(t.get_text()[:p])
    # g.set(xscale="log", yscale="log")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)

def load_incident(dataset_dir1, dataset_dir2):
    data = {}
    # Load raw data
    gt_file1 = np.load('../graphtransformer-main/data/'+dataset_dir1+'/graph_label_full_v2.npy') # 'eid', '5-min', 'rid', 'type_id', 'dists', 'dur (s)', 'q_len (m)'
    gt_file2 = np.load('../graphtransformer-main/data/' + dataset_dir2 + '/graph_label_full_v2.npy')
    # gt_file2[gt_file2[:, 3]>5, 3] = gt_file2[gt_file2[:, 3]>5, 3]+1
    # np.save('../graphtransformer-main/data/' + dataset_dir2 + '/graph_label_full_v2.npy', gt_file2, allow_pickle=True)
    gt1 = pd.DataFrame({'Dur-LA':gt_file1[:, -2]/60, 'Len-LA':gt_file1[:, -1]/1600, 'Type':gt_file1[:, 3], 'RID':gt_file1[:, 2], 'dists':gt_file1[:, 4]})
    gt2 = pd.DataFrame({'Dur-SB': gt_file2[:, -2] / 60, 'Len-SB': gt_file2[:, -1] / 1600, 'Type':gt_file2[:, 3], 'RID':gt_file2[:, 2], 'dists':gt_file2[:, 4]})
    # road_idx = np.load('../graphtransformer-main/data/'+dataset_dir+'/road_idx.npy')
    plt.rcParams["figure.figsize"] = (18, 18)
    plt.rcParams.update({'font.size': 20})
    # plt.ticklabel_format(style='sci')
    # fig, ax = plt.subplots(2, 3)
    # plot_hist(ax[0,0], 'Dur-LA', gt_file1[:, -2]/60, ylabel='# Events (LA)')
    # plot_hist(ax[1,0], 'Dur-SB', gt_file2[:, -2]/60, xlabel='Duration (min)', ylabel='# Events (SB)')
    # plot_hist(ax[0,1], 'Len-LA', gt_file1[:, -1]/1600)
    # plot_hist(ax[1,1], 'Len-SB', gt_file2[:, -1]/1600, xlabel='Length (mile)')
    # plot_hist(ax[0,2], 'Len-LA', gt_file1[:, -1][gt_file1[:, -1]>0]/1600)
    # plot_hist(ax[1,2], 'Len-SB', gt_file2[:, -1][gt_file2[:, -1]>0]/1600, xlabel='Length w/o zero (mile)')

    ## plot scatter by type
    # for i in np.sort(gt1['Type'].unique()):
    #     plot_scatter('LA', gt1[gt1['Type']==i], 'Type', xlabel='Duration (min)', ylabel='Len (mile)')
    #     plot_scatter('SB', gt2[gt2['Type']==i], 'Type', xlabel='Duration (min)', ylabel='Len (mile)')
    # plot_scatter('LA', gt1, 'Type', xlabel='Duration (min)', ylabel='Len (mile)')
    # plt.savefig('../graphtransformer-main/data/distribution_type_LA.png', bbox_inches = "tight")
    # plot_scatter('SB', gt2, 'Type', xlabel='Duration (min)', ylabel='Len (mile)')
    # plt.savefig('../graphtransformer-main/data/distribution_type_SB.png', bbox_inches = "tight")
    # plt.show()

    ## plot scatter by RID
    # plot_scatter('LA', gt1, 'RID', xlabel='Duration (min)', ylabel='Len (mile)')
    # plt.savefig('../graphtransformer-main/data/distribution_RID_LA.png', bbox_inches = "tight")
    # plot_scatter('SB', gt2, 'RID', xlabel='Duration (min)', ylabel='Len (mile)')
    # plt.savefig('../graphtransformer-main/data/distribution_RID_SB.png', bbox_inches = "tight")
    # plt.show()

    ## plot scatter by dist
    for r in np.sort(gt1['RID'].unique()):
        plot_scatter_continuous('LA', gt1[gt1['RID']==r], 'dists', xlabel='Duration (min)', ylabel='Len (mile)')
        plt.savefig('../graphtransformer-main/data/distribution_type_LA_'+str(r)+'.png')
        plt.show()
        # break

    for r in np.sort(gt2['RID'].unique()):
        plot_scatter_continuous('SB', gt2[gt2['RID']==r], 'dists', xlabel='Duration (min)', ylabel='Len (mile)')
        plt.savefig('../graphtransformer-main/data/distribution_type_SB_'+str(r)+'.png')
        plt.show()
        # break


    # gt=gt_file[:, 1:] # exclude event id
    # gt_m0, gt_m1 = gt[:, -2].max(), gt[:, -1].max()
    # for i in range(gt.shape[0]):
    #     r = gt[i, 1].astype(int)
    #     gt[i, 1] = np.where(road_idx==r)[0][0]
    # gt[:, -2], gt[:, -1] = gt[:, -2] / gt_m0, gt[:, -1] / gt_m1
    #
    # # rescale dists to [0, 1]
    # for r in range(len(road_idx)):
    #     r_event, r_sensor = gt[:, 1]==r, data['A_sr'][:, r]>0
    #     m_dist = np.maximum(gt[:, 3][r_event], sensor_dist[r_sensor]).max()
    #     gt[:, 3][r_event], sensor_dist[r_sensor] = gt[:, 3][r_event]/m_dist+r, sensor_dist[r_sensor]/m_dist+r # +r to distinguish roads
    #
    # data['sensor_dist'] = sensor_dist

dataset1, dataset2 = 'PEMS-07', 'PEMS-08'
dataloader = load_incident(dataset1, dataset2)



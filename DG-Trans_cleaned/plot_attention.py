import matplotlib
import numpy as np

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights

plt.rcParams['font.sans-serif'] = ['SimSun']  # set font family


def plot_attention(data, X_label=None, Y_label=None):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(40, 32))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)

    # Set axis labels
    if X_label != None and Y_label != None:
        #X_label = [x_label for x_label in X_label]
        #Y_label = [y_label for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=45)  # labels should be 'unicode'

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label, minor=False)  # labels should be 'unicode'
        ax.invert_yaxis()
        plt.colorbar(heatmap)

        ax.grid(True)

    # Save Figure
    plt.title(u'Attention Heatmap')
    file_name = './attention_heatmap_GT_2_rr.png'
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    # plt.show()
    plt.close(fig)  # close the figure


if __name__ == '__main__':
    asr, arr, ars = np.load('PEMSD8/PEMS-08asr_GT_2_sep.npy')[0], np.load('PEMSD8/PEMS-08arr_GT_2_sep.npy')[0], np.load('PEMSD8/PEMS-08ars_GT_2_sep.npy')[0]
    # atts = np.load('PEMSD8/PEMS-08atts_GT_2.npy')[0, :100, :100]
    x_label, y_label = [i for i in range(arr.shape[0])], [i for i in range(arr.shape[1])]
    plot_attention(arr, x_label, y_label)
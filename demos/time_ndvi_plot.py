import sys
import os
import numpy as np
from matplotlib import gridspec

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from load_model_data import *

def main():
    filename = None
    truth, context, target, npf = load_data_point()
    model = load_model()
    pred, _, _ = model(x = context, 
                       prediction_count = int((2/3)*truth.shape[-1]), 
                       non_pred_feat = npf)

    ndvi_truth = ((truth[:, 3, ...] - truth[ :, 2, ...]) / (
                truth[:, 3, ...] + truth[:, 2, ...] + 1e-6))
    cloud_mask = 1 - truth[:, 4, ...]
    ndvi_truth = ndvi_truth*cloud_mask

    ndvi_pred = ((pred[:, 3, ...] - pred[ :, 2, ...]) / (
                pred[:, 3, ...] + pred[:, 2, ...] + 1e-6))

    # take out cloudy days
    #splits = np.linspace(0.1,1,10)
    confidence = .5
    splits = [(1-confidence)/2, 0.5, 1 - (1-confidence)/2]
    q_t = np.quantile(ndvi_truth, splits, axis = (0,1,2))

    valid_ndvi_time = q_t[0]!=0
    q_t[0,valid_ndvi_time]

    x_t = np.arange(truth.shape[-1])[valid_ndvi_time]
    x_p = np.arange(truth.shape[-1] - pred.shape[-1], truth.shape[-1])

    q_p = np.quantile(ndvi_pred.detach().numpy(), splits, axis = (0,1,2))

    # plot pred and truth NDVI
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1]) 

    # the first subplot
    ax0 = plt.subplot(gs[0])

    ax0.plot(x_p, q_p[1,:], '--',color = 'b', label = 'pred. median')
    ax0.plot(x_t, q_t[1,valid_ndvi_time], '-',color = 'r', label = 'true median')

    ax0.legend(loc="upper left")

    ax0.fill_between(x_p, q_p[0,:],q_p[2,:], color = 'b', alpha=.1)
    ax0.fill_between(x_t, q_t[0,valid_ndvi_time],q_t[2,valid_ndvi_time], color = 'r', alpha=.1)

    # log scale for axis Y of the first subplot

    # the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax2 = plt.subplot(gs[2], sharex = ax0)

    ax0.grid()
    ax1.grid()
    ax2.grid()

    ax1.plot(np.arange(truth.shape[-1]-1)+1 , 50 * truth[0,6,0,0,:-1], label = 'precipitation')
    ax2.plot(np.arange(truth.shape[-1]-1)+1 , 50 *(2*truth[0,8,0,0,:-1] - 1), label = 'mean temp')

    ax0.set_ylabel("NDVI")
    ax1.set_ylabel("Rain" +"\n" + "(mm)")
    ax2.set_ylabel("Temp" +"\n" + "(°C)")
    ax2.set_xlabel("Time (x5 days)")

    plt.xlim([0, truth.size()[-1]])

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    '''
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)'''

    # put legend on first subplot
    #ax0.legend((line0, line1), ('red line', 'blue line'), loc='lower left')

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    if filename == None:
        plt.savefig('NDVI_time_series.png')
        plt.show()
    else:
        plt.savefig(filename)
    
    print("Done")

main()
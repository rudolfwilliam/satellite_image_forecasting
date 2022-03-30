"""Compare the first 35 training epochs of a run of SGDConvLSTM with zero baseline and 
   one with last frame baseline. Store pdf images of the plots for ENS and the different score components."""

import os
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt

file  = '/Data/lastframe_vs_zero.csv'
data = genfromtxt(os.getcwd() + file, delimiter=',')

score_components = ['ENS', 'SSIM', 'EMD', 'OLS', 'MAD']

# indexes of columns for zero and last frame baseline
idxs = [(5, -1), (4, 9), (1, 6), (2, 7), (3, 8)]

for comp, idx in zip(score_components, idxs):
    zero = data[1:36, idx[0]]
    lastframe = data[1:36, idx[1]]

    fig, ax = plt.subplots()

    zero, = ax.plot(zero, label='zero', color='r')
    lastframe, = ax.plot(lastframe, label='last frame', color='b')

    ax.legend((zero, lastframe), ('zero', 'last frame'), loc='lower right')
    plt.xlabel(comp)
    plt.ylabel('Epoch')

    # store image into visualizations directory by default
    plt.savefig(os.getcwd() + '/visualizations/' + comp + '.pdf') 
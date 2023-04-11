import sys
import os
import numpy as np
from matplotlib import gridspec
import datetime as dt
import copy
import argparse
import pickle
import torch
from scipy.interpolate import interp1d

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from load_model_data import *

dates = {
    "extreme": ['2018-01-28','2018-11-23'],
    "seasonal": ['2017-05-28','2020-04-11'],
    "ood": ['2017-07-02','2017-11-28'],
    "iid" : ['2017-06-20','2017-11-16'] 
}

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def main(index=0):

    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--index', type=int, default=0, help='index of datacube to use')
    parser.add_argument('-a', '--all', type=bool, default=False, help='whether to use all extreme samples')
    parser.add_argument('-t', '--tile', type=str, default=None, help='tile to use')
    args = parser.parse_args()

    #print("Evaluating on index " + str(args.index))
    #print("Evaluating on whole extreme dataset: " + str(args.all))

    files = os.listdir('demos/visualizations/ndvi_pickles')
    no_files = len(files)

    with open('demos/visualizations/ndvi_pickles/'+files[0], 'rb') as inp:
        data = pickle.load(inp)

    # These should be fixed
    x_t = data[0]
    x_p = data[2]

    q_t = data[1]
    for q in range(len(q_t)):
        q_t[q] = [0 for x in q_t[q]]

    q_ps = data[3]
    for i in range(len(q_ps)):
        q_ps[i] = np.zeros_like(q_ps[i])

    # Parse in the data
    for i in range(no_files):
        with open('demos/visualizations/ndvi_pickles/'+files[i], 'rb') as inp:
            data = pickle.load(inp)
        q_t += data[1]
        cur_q_ps = data[3]
        for j in range(len(q_ps)):
            q_ps[j] += cur_q_ps[j]

    # Divide by no samples
    q_t = [i/no_files for i in q_t]
    for i in range(len(q_ps)):
        q_ps[i] = [x/no_files for x in q_ps[i]]

    # Plotting itself
    model_names=["2019 weather","SGConvLSTM","SGEDConvLSTM"]
    colors = ["b","r","g","c","m","y"]
    colors = colors[:len(model_names)]

    fig, ax0 = plt.subplots()
    for q_p, mod_name, color in zip(q_ps, model_names, colors):
            ax0.plot(x_p, q_p[1], '--',color = color, label = mod_name)

    ax0.plot(x_t, q_t[1], '-', color = 'k', label = '2018 ground truth')

    ax0.legend(loc="upper right")
    #for q_p, color in zip(q_ps, colors):
    #    ax0.fill_between(x_p, q_p[0],q_p[2], color = color, alpha=.1)
    #ax0.fill_between(x_t, q_t[0],q_t[2], color = 'k', alpha=.1)

    #plt.axvline(dt.datetime.strptime("2018-06-05", '%Y-%m-%d'), color = "k", linestyle = ":")
    
    ax0.set_ylabel("NDVI (unitless)")
    ax0.set_xlabel("Time")

    #labels = ['Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct']
    #ax0.set_xticklabels(labels)

    days = [4, 32, 63, 93, 124, 154, 185, 216, 246, 277]
    days = [d/5 for d in days]
    plt.xticks(days, ['Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov'])

    plt.xlim([x_t[0],x_t[-1]])
    plt.ylim(0,1)

    plt.grid()

    plt.savefig('visualizations/final_ndvi.pdf', format="pdf")

if __name__ == "__main__":
    main()
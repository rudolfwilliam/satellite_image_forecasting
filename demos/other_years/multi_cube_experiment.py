import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import sample
import earthnet as en
from os import walk
import os
import datetime
import time
import warnings

from pandas import date_range
from get_coords import get_coords, get_limited_coords
from sentinel_download import get_data
import pickle

thr = .8

def main():
    warnings.simplefilter('once', RuntimeWarning)

    with open("demos/extreme_test_split_context_data_paths.pkl",'rb') as f:
        filenames = pickle.load(f)
    locations = []
    for f in filenames:
        if '32UNC' in f:
            locations.append(f[158:][:-4])
    """locations = ["32UQC_2018-01-28_2018-11-23_5305_5433_4665_4793_82_162_72_152",
                    "32UQC_2018-01-28_2018-11-23_5305_5433_4409_4537_82_162_68_148",
                    "32UQC_2018-01-28_2018-11-23_5305_5433_3257_3385_82_162_50_130",
                    "32UQC_2018-01-28_2018-11-23_5305_5433_4537_4665_82_162_70_150",
                    "32UQC_2018-01-28_2018-11-23_5305_5433_3385_3513_82_162_52_132"]"""
    N = 10
    locations = sample(locations,N)
    years = [[datetime.datetime(2016,1,28), datetime.datetime(2016,11,28)]]#,
             #[datetime.datetime(2017,1,28), datetime.datetime(2017,11,28)],
             #[datetime.datetime(2018,1,28), datetime.datetime(2018,11,28)],
             #[datetime.datetime(2019,1,28), datetime.datetime(2019,11,28)],
             #[datetime.datetime(2020,1,28), datetime.datetime(2020,11,28)],
             #[datetime.datetime(2021,1,28), datetime.datetime(2021,11,28)]]


    # Download
    years_data = []
    i = 0

    time_0 = time.time()
    for j, year in enumerate(years):
        data_locs = []
        for loc in locations:
            data_locs.append(get_data(get_coords(loc), start = year[0], end = year[1], n_chunks = 60))
            time_per_image = (time.time() - time_0) / (i + 1)
            print("downloaded {0}/{1} - ETA: {2} mins".format(i+1, len(locations)*len(years), time_per_image * ( len(locations)*len(years)  - i)/60))
            i += 1
        data_good = []
        for loc in data_locs:
            if loc.shape == (128,128,5,60):
                data_good.append(loc)
        years_data.append(np.stack(data_good, axis = 0))
        
    full_data = np.stack(years_data, axis = 0)


    '''full_data.shape:
        - years
        - location 
        - width
        - height
        - channel
        - time (within year)
    '''
    # NDVI computations
    full_data_ndvi = np.nan_to_num((full_data[:,:,:,:,3,:] - full_data[:,:,:,:,0,:]) / (full_data[:,:,:,:,3,:] + full_data[:,:,:,:,0,:]), 0)
    '''full_data_ndvi.shape:
        - years
        - location 
        - width
        - height
        - time (within year)
    '''
    full_data_mask = full_data[:,:,:,:,4,:] / 255
    full_data_ndvi_masked = np.ma.masked_array(full_data_ndvi, full_data_mask)
    # Quantiles
    splits = np.array([.25,.5,.75])
    axis_quantile = (1,2,3) # locations width height
    q = np.quantile(full_data_ndvi_masked, q = splits, axis = axis_quantile)
    valid_pixels = np.sum(1 - full_data_mask, axis = axis_quantile)
    ''' q.shape
        - splits
        - years
        - time
    '''
    np.savez_compressed("quantiles", q)
    np.save("valid_pixels", valid_pixels)
    valid_pixels_threshold = (valid_pixels > thr * np.max(valid_pixels)) & (q[2,...] != 0)

    # Date axis
    dates_bound = years[0] 
    dates = date_linspace(dates_bound[0],dates_bound[1], 60)

    # Plots
    Y = 3

    plt.plot(dates[valid_pixels_threshold[Y,:]], q[:, Y ,valid_pixels_threshold[Y,:]].T,'-*')
    plt.plot(dates, valid_pixels[Y, :]/(np.max(valid_pixels)) , ':*')

    plt.plot(dates, q[:, Y ,:].T,'-*')
    plt.plot(dates, valid_pixels[Y, :]/(np.max(valid_pixels)) , ':*')

    plt.show()

    # Plot against years
    for Y in range(len(years)):
        plt.plot(dates[valid_pixels_threshold[Y,:]], q[2, Y ,valid_pixels_threshold[Y,:]].T,'-*')
    plt.legend(["2016","2017","2018","2019","2020","2021"])
    plt.show()


def date_linspace(start, end, steps):
    delta = (end - start) / steps
    increments = range(0, steps) * np.array([delta]*steps)
    return start + increments

main()

import sys
import os
import numpy as np
from matplotlib import gridspec

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from load_model_data import *

def main():
    filename = None
    truth, context, target, npf = load_data_point(test_context_dataset = "Data/small_data/extreme_context_data_paths.pkl", 
                                                  test_target_dataset = "Data/small_data/extreme_target_data_paths.pkl")
    model = load_model()
    # no water 
    npf_no_water = copy.deepcopy(npf)
    npf_no_water[:,1,:,:,:] = 0*npf_no_water[:,1,:,:,:]
    preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = int((2/3)*truth.shape[-1]), 
                                        non_pred_feat = npf)
    preds_no_water, x_deltas, baselines = model(x = context, 
                                        prediction_count = int((2/3)*truth.shape[-1]), 
                                        non_pred_feat = npf_no_water)
    visualize_rgb([preds,preds_no_water], truth)
    print("Done")

def visualize_rgb(preds, truth, filename = None):
    """
    inputs:
        - preds is a list of predicted cubes (or only one)
        - truth is the ground truth
    """
    if not isinstance(preds, list):
        preds = [preds]
    pred_numpy = []
    for pred in preds:
        pred_numpy.append(pred.detach().numpy())
    preds = pred_numpy
    
    
    truth = truth.detach().numpy()
    # for smaller iid visualizations
    '''truth_idxs = [0,5,10,15,20,25]
    pred_idxs = [0,5,10,15]
    truth = np.take(truth, truth_idxs, axis=-1)
    pred = np.take(pred, pred_idxs, axis=-1)'''
    T = truth.shape[-1]
    t = T-pred.shape[-1]
    img = np.zeros((3, 128*T, 128*(1 + len(preds))))
    for i in range(T):
        img[:, 128*i: 128*(i + 1), 0:128] = truth[:, :3, :, :, i]
        if i >= t:
            for j in range(len(preds)):
                img[:, 128*i: 128*(i + 1),128*(j + 1):128*(j + 2)] = preds[j][:, :3, :, :, i - t]
    img = np.flip(img[:,:,:].astype(float),0)*2
    if filename == None:
        plt.imsave('visualizations/rgb.png', np.clip(img.transpose(1,2,0),0,1))
        plt.imsave('visualizations/rgb_landscape.png', np.clip(img.transpose(2,1,0),0,1))
        plt.show()
    else:
        plt.imsave(filename, np.clip(img.transpose(1,2,0),0,1))
    
    print("Done")

def visualize_ndvi(preds, truth, filename = None, gt = True):
    if not isinstance(preds, list):
        preds = [preds]
    pred_numpy = []
    for pred in preds:
        pred_numpy.append(pred.detach().numpy())
    preds = pred_numpy
    truth = truth.detach().numpy()

    ndvi_truth = ((truth[:, 3, ...] - truth[ :, 2, ...]) / (
                truth[:, 3, ...] + truth[:, 2, ...] + 1e-6))
    cloud_mask = 1 - truth[:, 4, ...]
    ndvi_truth = ndvi_truth*cloud_mask

    ndvi_preds = []

    for pred in preds:
        ndvi_preds.append(((pred[:, 3, ...] - pred[ :, 2, ...]) / (
                    pred[:, 3, ...] + pred[:, 2, ...] + 1e-6)))

    T = truth.shape[-1]
    t = int(T/3)
    img = np.zeros((128*T, 128*(1+len(preds))))
    for i in range(T):
        img[128*i: 128*(i + 1), 0:128] = ndvi_truth[0, :, :, i]
        if i >= t:
            for j in range(len(ndvi_preds)):
                img[128*i: 128*(i + 1),128*(j + 1):128*(j + 2)] = ndvi_preds[i][0, :, :, i - t]
    if filename == None:
        plt.imsave('visualizations/ndvi.png', np.clip(img,0,1))
        plt.show()
    else:
        plt.imsave(filename, np.clip(img,0,1))
    
    print("Done")

main()
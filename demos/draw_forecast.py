import sys
import os
import numpy as np
from matplotlib import gridspec

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from load_model_data import *

def main():
    filename = None
    truth, context, target, npf = load_data_point(test_context_dataset = "Data/portugal/seasonal_data_context_data_paths.pkl", 
                                                  test_target_dataset = "Data/portugal/seasonal_data_target_data_paths.pkl",
                                                  index = 100)
    model1 = load_model()
    #model2 = load_model("trained_models/model_epoch=031.ckpt")
    #pred2, _, _ = model2(x = context, 
    #                   prediction_count = int((2/3)*truth.shape[-1]), 
    #                   non_pred_feat = npf)
    
    pred1, _, _ = model1(x = context, 
                       prediction_count = int((2/3)*truth.shape[-1]), 
                       non_pred_feat = npf)
    #pred2, _, _ = model2(x = context, 
    #                   prediction_count = int((2/3)*truth.shape[-1]), 
    #                   non_pred_feat = npf)
    visualize_rgb([pred1], truth)#, undersample_indexs = [4,14,19,29,39,49,59])
    print("Done")

def visualize_rgb(preds, truth, filename = None, undersample_indexs = None):
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
    T = truth.shape[-1]
    t = T-pred.shape[-1]

    if undersample_indexs is not None:
        pred_idxs = []
        for idx in undersample_indexs:
            if idx >= int((1/3)*truth.shape[-1]):
                pred_idxs.append(idx - int((1/3)*truth.shape[-1]))
        truth = np.take(truth, undersample_indexs, axis=-1)
        pred_short = []
        for pred in preds:
            pred_short.append( np.take(pred, pred_idxs, axis=-1))
        preds = pred_short
        t = len(undersample_indexs)- len(pred_idxs)
        T = truth.shape[-1]
    
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

main()
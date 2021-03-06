import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime as dt

sys.path.append(os.getcwd())

from load_model_data import *

def main():
    mode = "extreme"
    index = 1
    filename = "demos/visualizations/forecast_{0}_{1}.pdf".format(mode, index)

    truth, context, _, npf = load_data_point(test_context_dataset = "Data/small_data/{0}_context_data_paths.pkl".format(mode), 
                                             test_target_dataset = "Data/small_data/{0}_target_data_paths.pkl".format(mode),
                                             index = index)
    # only use this with the complete dataset
    #truth, context, _, npf = load_data_point(test_context_dataset = "Data/{0}_data/{0}_data_context_data_paths.pkl".format(mode), 
    #                                         test_target_dataset = "Data/{0}_data/{0}_data_target_data_paths.pkl".format(mode),
    #                                         index = index)
    SGConvLSTM = load_model("trained_models/SGConvLSTM.ckpt")
    SGEDConvLSTM = load_model("trained_models/SGEDConvLSTM.ckpt")

    dates = {
        "extreme": ['2018-01-28','2018-11-23'],
        "seasonal": ['2017-05-28','2020-04-11'],
        "ood": ['2017-07-02','2017-11-28'],
        "iid" : ['2017-06-20','2017-11-16'] 
    }
    #pred2, _, _ = model2(x = context, 
    #                   prediction_count = int((2/3)*truth.shape[-1]), 
    #                   non_pred_feat = npf)
    # No water 
    #npf_no_water = copy.deepcopy(npf)
    #npf_no_water[:,1,:,:,:] = 0*npf_no_water[:,1,:,:,:]

    pred1, _, _ = SGConvLSTM(x = context, 
                             prediction_count = int((2/3)*truth.shape[-1]), 
                             non_pred_feat = npf)
    pred2, _, _ = SGEDConvLSTM(x = context, 
                               prediction_count = int((2/3)*truth.shape[-1]), 
                               non_pred_feat = npf)

    visualize_rgb([pred1,pred2], 
                  truth, 
                  filename = filename, 
                  labels = ["SGConvLSTM","SGEDConvLSTM"],
                  undersample_indexs = [4,14,20,29,38,51,59],
                  dates_bounds = dates[mode])
    print("Done")

def visualize_rgb(preds, truth, filename = None, undersample_indexs = None, labels = None, dates_bounds = None, draw_axis = True):
    """
    inputs:
        - preds is a list of predicted cubes (or only one)
        - truth is the ground truth
    """
    if labels == None:
        labels = ["Ground Truth"] + [""]*len(preds)
    else:
        labels = ["Ground Truth"] + labels

    T0 = truth.shape[-1]

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

    if dates_bounds is not None:
        dates_b = [dt.datetime.strptime(dates_bounds[0], '%Y-%m-%d'),dt.datetime.strptime(dates_bounds[1], '%Y-%m-%d')] 
        dates = date_linspace(dates_b[0],dates_b[1],T0)
        undersample_dates = dates[undersample_indexs]
        dates_strings = []
        for date in undersample_dates:
            dates_strings.append(date.strftime("%d %b"))
        plt.figure(figsize=(10, 5))
        plt.imshow(np.clip(img.transpose(2,1,0),0,1)*2)
        plt.yticks((128/2)+(np.arange(3)*128), labels,rotation='vertical',horizontalalignment="right", verticalalignment="center")
        plt.tick_params(axis='both', which='both',length=0)
        plt.xticks((128/2)+(np.arange(len(undersample_indexs))*128),dates_strings)        
        if filename is None:
            plt.plot()
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
    elif dates_bounds is None and filename is not None:
        plt.imshow(np.clip(img.transpose(2,1,0),0,1)*2)
        plt.yticks((128/2)+(np.arange(3)*128), labels,rotation='vertical',horizontalalignment="right", verticalalignment="center")
        plt.tick_params(axis='both', which='both',length=0)
        plt.xticks((128/2)+(np.arange(len(undersample_indexs))*128),undersample_indexs)
        if filename is None:
            plt.plot()
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

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
        plt.imshow(np.clip(img,0,1))
        plt.show()
    else:
        plt.imsave(filename, np.clip(img,0,1))

def date_linspace(start, end, steps):
    delta = (end - start) / steps
    increments = range(0, steps) * np.array([delta]*steps)
    return start + increments

if __name__ == "__main__":
    main()
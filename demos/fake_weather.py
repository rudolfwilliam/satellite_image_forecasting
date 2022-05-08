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
                                                  test_target_dataset = "Data/small_data/extreme_target_data_paths.pkl",
                                                  index = 0)
    
    model1 = load_model()
    model2 = load_model("trained_models/top_performant_autoenc.ckpt")
    #pred2, _, _ = model2(x = context, 
    #                   prediction_count = int((2/3)*truth.shape[-1]), 
    #                   non_pred_feat = npf)
    # No water 
    #npf_no_water = copy.deepcopy(npf)
    #npf_no_water[:,1,:,:,:] = 0*npf_no_water[:,1,:,:,:]

    truth_19, context_19, target_19, npf_19 = load_data_point(train_dataset = "Data/ger_data/train_data_paths.pkl",
                                            index=0)
    npf_19 = truth_19[:,5:,...]
    truth_modified = copy.deepcopy(truth)
    truth_modified[:, 5:10, :, :, 12:(12+30)] = truth_19[:,5:10,...]
    npf_modified = truth_modified[:, 5:, :, :, 20:]
    context_modified = truth_modified[:, :, :, :, :20]
    """npf_modified = copy.deepcopy(npf)
    max_rain = (npf_modified[:,1,:,:,:]).max()
    npf_modified[:,1,:,:,:] = 0.5 * max_rain * np.ones((npf_modified[:,1,:,:,:]).shape)
    """ 
    pred1, _, _ = model1(x = context, 
                       prediction_count = int((2/3)*truth.shape[-1]), 
                       non_pred_feat = npf)
    pred_mo, _, _ = model1(x = context_modified, 
                    prediction_count = int((2/3)*truth.shape[-1]), 
                    non_pred_feat = npf_modified)
    pred2, _, _ = model2(x = context, 
                       prediction_count = int((2/3)*truth.shape[-1]), 
                       non_pred_feat = npf)
    visualize_rgb([pred_mo, pred1,pred2], truth,filename="demos/visualizations/extreme_forcast.pdf", draw_axis=False)
    print("Done")

def fake_weather(preds, truth, filename = None, undersample_indexs = None, factor = 2, draw_axis = True):
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

    # 25
    if filename == None:
        filename = 'demos/visualizations/rgb.png'

    i = 25
    
    img_to_plot = np.clip(factor* img.transpose(2,1,0),0,1)
    img_to_plot = img_to_plot[:(3*128),i*128:(i+1)*128,:]
    final = np.zeros((128,3*128,3))

    final[:,0*128:(0+1)*128,:] = img_to_plot[0*128:(0+1)*128,:,:]
    final[:,1*128:(1+1)*128,:] = img_to_plot[2*128:(2+1)*128,:,:]
    final[:,2*128:(2+1)*128,:] = img_to_plot[1*128:(1+1)*128,:,:]
    fig, ax = plt.subplots(1, 1)
    ax.tick_params(axis='x', which='both', length=0)
    ax.imshow(final)
    ax.xaxis.tick_top()
    plt.yticks([])
    plt.xticks(64 + 128*np.arange(3),labels = ["Ground Truth\n","SGConvLSTM\n(real weather)","SGConvLSTM\n(2019 'replaced' weather)"] )

    plt.savefig("alt_25.pdf", bbox_inches = "tight")
    plt.imsave(filename, img_to_plot)

    '''else:
        plt.figure(figsize=(10, 5), dpi=600)
        plt.imshow(img_to_plot)
        plt.yticks([64, 64+128, 64+128+128], labels = ["Ground truth","SGConvLSTM","SGEDConvLSTM"],rotation='vertical',va='center')
        plt.xticks(64 + 128*np.arange(len(undersample_indexs)),labels = np.array(undersample_indexs)+1 )
        plt.tick_params(axis='y', which='both', length=0)
        plt.xlabel("Time")
        plt.savefig(filename, bbox_inches = "tight")'''


main()
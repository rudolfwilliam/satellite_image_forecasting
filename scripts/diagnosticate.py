import sys
import os
from os.path import join
import numpy as np
import pickle
from matplotlib.widgets import Slider, RadioButtons
from matplotlib import gridspec
import copy

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from config.config import diagnosticate_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from drought_impact_forecasting.models.Conv_model import Conv_model
from drought_impact_forecasting.models.Peephole_LSTM_model import Peephole_LSTM_model
from Data.data_preparation import Earthnet_Dataset, Earthnet_Test_Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():

    cfg = diagnosticate_line_parser()
    mesoscale_cut = [39, 41]

    print("Diagnosticating experiment {0}".format(cfg['run_name']))
    print("Diagnosticating model at epoch {0}".format(cfg['epoch_to_validate']))
    
    if cfg['train_data'] is not None:
        with open(join(os.getcwd(), cfg['train_data']),'rb') as f:
            training_path_list = pickle.load(f)
        dataset = Earthnet_Dataset(training_path_list, mesoscale_cut)
    else:
        with open(join(os.getcwd(), cfg['test_context_data']),'rb') as f:
            test_context_path_list = pickle.load(f)
        with open(join(os.getcwd(), cfg['test_target_data']),'rb') as f:
            test_target_path_list = pickle.load(f)
        dataset = Earthnet_Test_Dataset(test_context_path_list, test_target_path_list, mesoscale_cut)

    model = Peephole_LSTM_model.load_from_checkpoint(cfg['model_path'])
    model.eval()
    
    truth = dataset.__getitem__(cfg['index'])

    truth = truth.unsqueeze(dim=0)
    T = truth.shape[-1]
    t0 = int(T/3)
    context = truth[:, :, :, :, :t0] # b, c, h, w, t
    target = truth[:, :5, :, :, t0:] # b, c, h, w, t
    npf = truth[:, 5:, :, :, t0:]

    x_preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = T-t0, 
                                        non_pred_feat = npf)
    
    if cfg['action'] == 'visualize':
        generate_plot(x_preds, truth)
    elif cfg['action'] == 'time_plot':
        plot_time(x_preds, truth)
    elif cfg['action'] == 'visualize_in_time':
        visualize_rgb(x_preds, truth)
    elif cfg['action'] == 'visualize_in_time_ndvi':
        visualize_ndvi(x_preds, truth)
    elif cfg['action'] == 'synthetic weather':
        fake_weather(model, truth, t0, T)
    '''real_deltas = target[:, :4, ...] - truth[:,:4,:,:,t0-1:-1]
    full_mit_baselines = truth
    full_mit_baselines[:, :4, :, :, t0:] = real_deltas
    generate_plot(x_deltas, full_mit_baselines)'''

def fake_weather(model, truth, t0, T):
    path = "visualizations/fake_weather"
    context = truth[:, :, :, :, :t0] # b, c, h, w, t
    target = truth[:, :5, :, :, t0:] # b, c, h, w, t
    npf = truth[:, 5:, :, :, t0:]
    temp_truth = copy.deepcopy(truth)
    #Original
    x_preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = T-t0, 
                                        non_pred_feat = npf)

    visualize_rgb(x_preds, truth, path + "/original.png")
    visualize_ndvi(x_preds, truth, path + "/original_ndvi.png")
    plot_time(x_preds, truth, path + "/original_time.png")
    #No water 
    npf_no_water = copy.deepcopy(npf)
    npf_no_water[:,1,:,:,:] = 0*npf_no_water[:,1,:,:,:]
    x_preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = T-t0, 
                                        non_pred_feat = npf_no_water)
    temp_truth[:, 5:, :, :, t0:] = npf_no_water
    visualize_rgb(x_preds, temp_truth, path + "/no_water.png", False)
    visualize_ndvi(x_preds, temp_truth, path + "/no_water_ndvi.png", False)
    plot_time(x_preds, temp_truth, path + "/no_water_time.png")

    #always_rain
    npf_always_rain = copy.deepcopy(npf)
    npf_always_rain[:,1,:,:,:] = 0.1+0*npf_always_rain[:,1,:,:,:]
    x_preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = T-t0, 
                                        non_pred_feat = npf_always_rain)

    temp_truth[:, 5:, :, :, t0:] = npf_always_rain
    visualize_rgb(x_preds, temp_truth, "visualizations/fake_weather/too_always_rain.png", False)
    visualize_ndvi(x_preds, temp_truth, "visualizations/fake_weather/too_always_rain_ndvi.png", False)
    plot_time(x_preds, temp_truth, "visualizations/fake_weather/too_always_rain_time.png")

    #stronger_rain
    npf_stronger_rain = copy.deepcopy(npf)
    npf_stronger_rain[:,1,:,:,:] = 3*npf_stronger_rain[:,1,:,:,:]
    x_preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = T-t0, 
                                        non_pred_feat = npf_stronger_rain)

    temp_truth[:, 5:, :, :, t0:] = npf_stronger_rain
    visualize_rgb(x_preds, temp_truth, "visualizations/fake_weather/stronger_rain.png",False)
    visualize_ndvi(x_preds, temp_truth, "visualizations/fake_weather/stronger_rain_ndvi.png",False)
    plot_time(x_preds, temp_truth, "visualizations/fake_weather/stronger_rain_time.png")
    pass

def visualize_rgb(pred, truth, filename = None, gt = True):
    pred = pred.detach().numpy()
    truth = truth.detach().numpy()

    # For smaller iid visualizations
    '''truth_idxs = [0,5,10,15,20,25]
    pred_idxs = [0,5,10,15]
    truth = np.take(truth, truth_idxs, axis=-1)
    pred = np.take(pred, pred_idxs, axis=-1)'''
    if gt:
        T = truth.shape[-1]
        t = T-pred.shape[-1]
        img = np.zeros((3, 128*T, 128*2))
        for i in range(T):
            img[:, 128*i: 128*(i + 1), 0:128] = truth[:, :3, :, :, i]
            if i >= t:
                img[:, 128*i: 128*(i + 1),128:256] = pred[:, :3, :, :, i - t]
        img = np.flip(img[:,:,:].astype(float),0)*2
    else:
        T = truth.shape[-1]
        t = T-pred.shape[-1]
        img = np.zeros((3, 128*T, 128))
        for i in range(T):
            if i >= t:
                img[:, 128*i: 128*(i + 1),:128] = pred[:, :3, :, :, i - t]
        img = np.flip(img[:,:,:].astype(float),0)*2
    if filename == None:
        plt.imsave('rgb.png', np.clip(img.transpose(1,2,0),0,1))
        plt.imsave('rgb_landscape.png', np.clip(img.transpose(2,1,0),0,1))
        plt.show()
    else:
        plt.imsave(filename, np.clip(img.transpose(1,2,0),0,1))
    
    print("Done")

def visualize_ndvi(pred, truth, filename = None, gt = True):
    pred = pred.detach().numpy()
    truth = truth.detach().numpy()
    ndvi_truth = ((truth[:, 3, ...] - truth[ :, 2, ...]) / (
                truth[:, 3, ...] + truth[:, 2, ...] + 1e-6))
    cloud_mask = 1 - truth[:, 4, ...]
    ndvi_truth = ndvi_truth*cloud_mask
    ndvi_pred = ((pred[:, 3, ...] - pred[ :, 2, ...]) / (
                pred[:, 3, ...] + pred[:, 2, ...] + 1e-6))


    T = truth.shape[-1]
    t = int(T/3)
    if gt:
        img = np.zeros((128*T, 128*2))
        for i in range(T):
            img[128*i: 128*(i + 1), 0:128] = ndvi_truth[0, :, :, i]
            if i >= t:
                img[128*i: 128*(i + 1),128:256] = ndvi_pred[0, :, :, i - t]
    else:
        img = np.zeros((128*T, 128))
        for i in range(T):
            if i >= t:
                img[128*i: 128*(i + 1),:128] = ndvi_pred[0, :, :, i - t]
    if filename == None:
        plt.imsave('ndvi.png', np.clip(img,0,1))
        plt.show()
    else:
        plt.imsave(filename, np.clip(img,0,1))
    
    print("Done")

def plot_time(pred, truth, filename = None):
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

def generate_plot(pred, true):
    T = true.shape[-1]
    t = int(T/3)
    channel = 0

    im1, im2, im3 = get_image(pred, true, channel, t)
    fig, ax = plt.subplots(ncols=5,nrows=1, gridspec_kw={'width_ratios': [1, 1, 1,.8,.2]})
    ax[0].set_title('Truth')
    ax[1].set_title('Prediction')
    ax[2].set_title('Delta')
    ims1 = ax[0].imshow(im1)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("bottom", size="10%", pad=0.05)
    cbar1 = plt.colorbar(ims1, cax=cax1, orientation='horizontal')

    if im2 is not None:
        ims2 = ax[1].imshow(im2)
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("bottom", size="10%", pad=0.05)
        cbar2 = plt.colorbar(ims2, cax=cax2, orientation='horizontal')
    if im3 is not None:
        ims3 = ax[2].imshow(im3)
        divider3 = make_axes_locatable(ax[2])
        cax3 = divider3.append_axes("bottom", size="10%", pad=0.05)
        cbar3 = plt.colorbar(ims3, cax=cax3, orientation='horizontal')
    
    radio = RadioButtons(ax[3], ('G', 'B', 'R', 'I', 'mask', 'elev', 'prec', 'pres', 'mean T', 'min T', 'max T'))
    time_silder = Slider(
        ax=ax[4],
        label='time',
        valmin=0,
        valmax=T-0.01,
        valinit=t,
        orientation="vertical"
    )

    def update(val):
        nonlocal t
        t = int(np.floor(time_silder.val))
        #print("channel: {0} - time: {1}".format(channel,t))
        im1, im2, im3 = get_image(pred, true, channel, t)
        ims1.set_data(im1)
        cbar1.update_normal(ims1)
        if im2 is not None:
            ims2.set_data(im2)  
            cbar2.update_normal(ims2)
        if im3 is not None:
            ims3.set_data(im3)
            cbar3.update_normal(ims3)
        fig.canvas.draw_idle()
        
    def update_radio(val):
        channel_dict = {'G':0, 'B':1, 'R':2, 'I':3, 'mask':4, 'elev':5, 'prec':6, 'pres':7, 'mean T':8, 'min T':9, 'max T':10}
        nonlocal channel
        channel = channel_dict[val]
        #print("channel: {0} - time: {1}".format(channel,t))
        im1, im2, im3 = get_image(pred, true, channel, t)
        ims1.set_data(im1)
        ims1.set_clim(im1.min(),im1.max())
        cbar1.update_normal(ims1)
        if im2 is not None:
            ims2.set_data(im2)  
            ims2.set_clim(im2.min(),im2.max())
            cbar2.update_normal(ims2)
        if im3 is not None:
            ims3.set_data(im3)
            ims3.set_clim(im3.min(),im3.max())
            cbar3.update_normal(ims3)
        fig.canvas.draw_idle()

    radio.on_clicked(update_radio)
    time_silder.on_changed(update)
    
    plt.show()
    print("Done")
    
def get_image(pred, true, channel, time):
    T = true.shape[-1]
    t0 = T - pred.shape[-1]
    im_true = true[0,channel,:,:,time]
    if time >= t0 and channel < 4:
        im_pred = pred[0,channel,:,:,time - t0]
        im_delt = im_true - im_pred
        return im_true.cpu().detach().numpy(), im_pred.cpu().detach().numpy(), im_delt.cpu().detach().numpy()
    else:
        return im_true.cpu().detach().numpy(), None, None

if __name__ == "__main__":
    main()
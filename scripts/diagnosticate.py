import sys
import os
from os.path import join
import numpy as np
import pickle
from matplotlib.widgets import Slider, RadioButtons

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
    '''real_deltas = target[:, :4, ...] - truth[:,:4,:,:,t0-1:-1]
    full_mit_baselines = truth
    full_mit_baselines[:, :4, :, :, t0:] = real_deltas
    generate_plot(x_deltas, full_mit_baselines)'''

def plot_time(pred, truth):
    ndvi_truth = ((truth[:, 3, ...] - truth[ :, 2, ...]) / (
                truth[:, 3, ...] + truth[:, 2, ...] + 1e-6))
    cloud_mask = 1 - truth[:, 4, ...]
    ndvi_truth = ndvi_truth*cloud_mask

    ndvi_pred = ((pred[:, 3, ...] - pred[ :, 2, ...]) / (
                pred[:, 3, ...] + pred[:, 2, ...] + 1e-6))

    # Take out cloudy days
    #splits = np.linspace(0.1,1,10)
    splits = [0.25, 0.5, 0.75]
    q_t = np.quantile(ndvi_truth, splits, axis = (0,1,2))

    valid_ndvi_time = q_t[0]!=0
    q_t[0,valid_ndvi_time]

    x_t = np.arange(truth.shape[-1])[valid_ndvi_time]
    x_p = np.arange(truth.shape[-1] - pred.shape[-1],truth.shape[-1])

    q_p = np.quantile(ndvi_pred.detach().numpy(), splits, axis = (0,1,2))

    # Plot pred and truth NDVI
    for i in range(len(splits)):
        plt.plot(x_p, q_p[i,:], '--*', label = 'truth ' + str(splits[i]))
        plt.plot(x_t, q_t[i,valid_ndvi_time], '-*', label = 'pred ' + str(splits[i]))

    # Plot weather
    plt.plot(np.arange(truth.shape[-1] - 1), truth[0,6,0,0,:-1], label = 'precipitation')
    plt.plot(np.arange(truth.shape[-1] - 1), truth[0,8,0,0,:-1], label = 'mean temp')
    #plt.plot(np.arange(truth.shape[-1] - 1), truth[0,9,0,0,:-1], label = 'min temp')
    #plt.plot(np.arange(truth.shape[-1] - 1), truth[0,10,0,0,:-1], label = 'max temp')
    
    plt.title("NDVI Time series")
    plt.ylabel("NDVI/Weather")
    plt.xlabel("Time")


    plt.legend(loc='upper left', title="Time NDVI")
    plt.show()
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
        valmax=T,
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
    im_truu = true[0,channel,:,:,time]
    if time >= t0 and channel < 4:
        im_pred = pred[0,channel,:,:,time - t0]
        im_delt = im_truu - im_pred
        return im_truu.cpu().detach().numpy(), im_pred.cpu().detach().numpy(), im_delt.cpu().detach().numpy()
    else:
        return im_truu.cpu().detach().numpy(), None, None

if __name__ == "__main__":
    main()
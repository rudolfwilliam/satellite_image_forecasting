from logging import Logger
import sys
import os
import numpy as np
import random
from shutil import copy2
from os import listdir
from matplotlib.widgets import Slider, RadioButtons
import pickle
#from pytorch_lightning.accelerators import acceleratofrom pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config.config import command_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from drought_impact_forecasting.models.Conv_model import Conv_model
from Data.data_preparation import Earthnet_Dataset, prepare_test_data
from scripts.callbacks import WandbTest_callback
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wandb
from datetime import datetime



def main():
    args, cfg = command_line_parser(mode = 'validate')

    #filepath = os.getcwd() + cfg["project"]["model_path"]
    model_path = os.path.join(cfg['path_dir'], "files", "runtime_model")
    models = listdir(model_path)
    models.sort()
    model_path = os.path.join(model_path , models[args.me])
    # to check that it's the last model

    print("diagnosting experiment {0}".format(args.rn))
    print("diagnosting model at epoch {0}".format(args.me))

    #GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("GPU count: {0}".format(gpu_count))

    with open(os.path.join(cfg['path_dir'], "files", "val_2_data_paths.pkl"),'rb') as f:
        val_2_path_list = pickle.load(f)
    
    val_2_data = Earthnet_Dataset(val_2_path_list, cfg["data"]["mesoscale_cut"], device=device)

    truth = val_2_data.__getitem__(np.random.choice(range(val_2_data.__len__())))

    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    elif args.model_name == "Conv_model":
        model = Conv_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise ValueError("The specified model name is invalid.")
    truth = truth.unsqueeze(dim=0)
    T = truth.shape[-1]
    t0 = 20
    context = truth[:, :, :, :, :t0] # b, c, h, w, t
    target = truth[:, :5, :, :, t0:] # b, c, h, w, t
    npf = truth[:, 5:, :, :, t0+1:]

    x_preds, x_deltas, baselines = model(x = context, 
                                        prediction_count = T-t0, 
                                        non_pred_feat = npf)
    generate_plot(x_preds.permute(1,2,3,4,0), truth)

def generate_plot(pred, true):
    T = true.shape[-1]
    t = 25
    channel = 0

    im1, im2, im3 = get_image(pred, true, channel, t)
    fig, ax = plt.subplots(ncols=5,nrows=1, gridspec_kw={'width_ratios': [1, 1, 1,.2,.2]})
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
    
    radio = RadioButtons(ax[3], ('G', 'B', 'R', 'I', 'mask', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6'))
    time_silder = Slider(
        ax=ax[4],
        label='time',
        valmin=0,
        valmax=T - 1,
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
        channel_dict = {'G':0, 'B':1, 'R':2, 'I':3, 'mask':4, 'w1':5, 'w2':6, 'w3':7, 'w4':8, 'w5':9, 'w6':10}
        nonlocal channel
        channel = channel_dict[val]
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

    radio.on_clicked(update_radio)
    time_silder.on_changed(update)
    
    plt.show()
    print("Done")

    pass
    
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
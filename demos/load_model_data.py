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
from drought_impact_forecasting.models.EN_model import EN_model 
from Data.data_preparation import Earthnet_Dataset, Earthnet_Test_Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_model(model_path = "trained_model/top_performant.ckpt"):
    """
        load a model from file for inference
    """
    model = EN_model.load_from_checkpoint(model_path)
    model.eval()
    return model

def load_data_point(train_dataset = None, 
                    test_context_dataset = None, 
                    test_target_dataset = None,
                    index = 0):
    """
        load a data cube from file to the common used structure. You can either pick a datapoint from a training dataset or a test dataset. 
        Parameters:
            train_dataset: path to a pickle files that idicates a training dataset
            test_context_dataset: path to a pickle files that idicates the contexts of a test dataset
            test_target_dataset: path to a pickle files that idicates the targets of a test dataset
        Such pickles can be easly generated with data_collection.py
        Note: you should either use train_dataset or both test_context_dataset and test_target_dataset.
        Outputs:
            truth: a full cube 
            context: the part of the cube that can be used for context
            target: the part of the cube that must be forecasted
            npf: non predictive features (weather) in the future
    """
    # Parameter checking
    if train_dataset is not None and (test_context_dataset is not None or test_context_dataset is not None): 
        raise ValueError("You can either use data from training dataset or from testing dataset, not both")
    if (test_context_dataset is not None and test_target_dataset is None) or (test_context_dataset is None and test_target_dataset is not None): 
        raise ValueError("When using test dataset both context and target must be specified")
    
    # train_dataset case
    if train_dataset is not None:
        with open(join(os.getcwd(), train_dataset),'rb') as f:
            training_path_list = pickle.load(f)
        dataset = Earthnet_Dataset(training_path_list, [39,41])
    # test_dataset case
    else:
        with open(join(os.getcwd(), test_context_dataset),'rb') as f:
            test_context_path_list = pickle.load(f)
        with open(join(os.getcwd(), test_target_dataset),'rb') as f:
            test_target_path_list = pickle.load(f)
        dataset = Earthnet_Test_Dataset(test_context_path_list, test_target_path_list, [39,41])
    # get cube
    truth = dataset.__getitem__(index)
    # cube splitting
    truth = truth.unsqueeze(dim=0)
    T = truth.shape[-1]
    t0 = int(T/3)
    context = truth[:, :, :, :, :t0] # b, c, h, w, t
    target = truth[:, :5, :, :, t0:] # b, c, h, w, t
    npf = truth[:, 5:, :, :, t0:]
    return truth, context, target, npf
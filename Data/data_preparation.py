import earthnet as en
import numpy as np
import torch
import os
from os.path import isfile, join

def prepare_data(train):
    # data = np.load('../Data/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz')
    # data = np.load(os.getcwd() + '/Data/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz')
    if train:
        data_dir = os.getcwd() + '/Data/train/'
    else:
        data_dir = os.getcwd() + '/Data/test/'
    
    # Load all datacubes from data folder
    #data = [np.load(data_dir + f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    data = np.load(os.getcwd() + '/Data/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz')
    dataset = Earthnet_Dataset(data)
    return dataset


class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Only works for a single training instance
        #self.data = 
        self.highres_dynamic_context = np.expand_dims(data['highresdynamic'][:, :, :, :10], axis=0)
        self.highres_static = np.expand_dims(data['highresstatic'], axis=0)
        self.meso_dynamic = np.expand_dims(data['mesodynamic'], axis=0)
        self.meso_static = np.expand_dims(data['mesostatic'], axis=0)

        # 'Label' only consists of the future satellite images
        self.highres_dynamic_target = np.expand_dims(data['highresdynamic'][:, :, :, 10:], axis=0)

    def __len__(self):
        return self.highres_dynamic_context.shape[0]

    def __getitem__(self, index):
        return self.highres_dynamic_context[index], self.highres_static[index], self.meso_dynamic[index], \
               self.meso_static[index], self.highres_dynamic_target[index]



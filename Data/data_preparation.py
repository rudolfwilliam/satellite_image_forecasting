import earthnet as en
import numpy as np
import torch

def prepare_data():
    data = np.load('../Data/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz')
    dataset = Earthnet_Dataset(data)
    return dataset

class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Only works for a single training instance
        self.context = np.expand_dims(np.array([('highresdynamic', data['highresdynamic'][:, :, :, :10]),
                                               ('highresstatic', data['highresstatic']),
                                               ('mesodynamic', data['mesodynamic']),
                                               ('mesostatic', data['mesostatic'])]), axis = 0)

        # 'Label' only consists of the future satellite images
        self.target = np.expand_dims(data['highresdynamic'][:, :, :, 10:], axis=0)

    def __len__(self):
        return self.context.shape[0]

    def __getitem__(self, index):
        return self.context[index], self.target[index]



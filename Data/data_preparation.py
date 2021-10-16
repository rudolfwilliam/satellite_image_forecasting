import earthnet as en
import numpy as np
import torch
import os
from os.path import isfile, join

def prepare_data(train_dir = '/Data/train/', test_dir = '/Data/test'):

    '''train_files = []
    for path, subdirs, files in os.walk(os.getcwd() + train_dir):
        for name in files:
            train_files.append(os.path.join(path, name))
    print("train")
    print(train_files)
 
    test_files = []
    for path, subdirs, files in os.walk(os.getcwd() + test_dir):
        for name in files:
            test_files.append(os.path.join(path, name))
    print("test")
    print(test_files)'''

    # Load all datacubes from data folder
    #data = [np.load(data_dir + f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    data = np.load(os.getcwd() + '/Data/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz')
    dataset = Earthnet_Dataset(data)
    return dataset, dataset


class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Only works for a single training instance
        self.highres_dynamic = np.expand_dims(data['highresdynamic'], axis=0)
        self.highres_static = np.expand_dims(data['highresstatic'], axis=0)
        self.meso_dynamic = np.expand_dims(data['mesodynamic'], axis=0)
        self.meso_static = np.expand_dims(data['mesostatic'], axis=0)

        ''' Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        print(torch.Tensor(self.highres_dynamic).size())
        self.highres_dynamic = torch.Tensor(self.highres_dynamic).permute(0, 3, 1, 2, 4)
        # 'Label' only consists of the future satellite images
        #self.highres_dynamic_target = np.expand_dims(data['highresdynamic'][:, :, :, 10:], axis=0)

    def __len__(self):
        return self.highres_dynamic.shape[0]

    def __getitem__(self, index):
        return self.highres_dynamic[index], self.highres_static[index], self.meso_dynamic[index], \
               self.meso_static[index]

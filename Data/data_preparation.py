import earthnet as en
import numpy as np
import torch
import os
from os.path import isfile, join
 
def prepare_data(training_samples, train_dir = '/Data/train/', test_dir = '/Data/test'):
   
    train_files = []
    for path, subdirs, files in os.walk(os.getcwd() + train_dir):
        for name in files:
            train_files.append(join(path, name))
    
    test_context_files = []
    test_target_files = []
    for path, subdirs, files in os.walk(os.getcwd() + test_dir):
        for name in files:
            full_name = join(path, name)
            if 'context' in full_name:
                test_context_files.append(full_name)
            else:
                test_target_files.append(full_name)

    # Sort file names just in case (so we glue together the right context & target)
    test_context_files.sort()
    test_target_files.sort()
 
    train_data = []
    test_context_data = []
    test_target_data = []
    
    # limit number of training samples
    for i in range(min(len(train_files), training_samples)):
        train_data.append(np.load(train_files[i]))
 
    for i in range(len(test_context_files)):
        test_context_data.append(np.load(test_context_files[i]))
        test_target_data.append(np.load(test_context_files[i]))
    
    train = Earthnet_Dataset(train_data)
    test = Earthnet_Dataset(test_context_data, test_target_data)
    return train, test
 
 
class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, context, target = None):
        
        samples = len(context)

        # If target data is given separately add context + target dimensions
        if target is not None:
            hrs_shape = list(context[0]['highresdynamic'].shape)
            hrs_shape[-1] += target[0]['highresdynamic'].shape[-1]
            self.highres_dynamic = np.empty((tuple([samples] + hrs_shape)))
        else:
            self.highres_dynamic = np.empty((tuple([samples] + list(context[0]['highresdynamic'].shape))))
        self.highres_static = np.empty((tuple([samples] + list(context[0]['highresstatic'].shape))))
        self.meso_dynamic = np.empty((tuple([samples] + list(context[0]['mesodynamic'].shape))))
        self.meso_static = np.empty((tuple([samples] + list(context[0]['mesostatic'].shape))))

        for i in range(samples):
            # For test samples glue together context & target
            if target is not None:
                self.highres_dynamic[i] = np.append(context[i]['highresdynamic'], target[i]['highresdynamic'],axis=-1)
            else:
                self.highres_dynamic[i] = context[i]['highresdynamic']
            self.highres_static[i] = context[i]['highresstatic']
            self.meso_dynamic[i] = context[i]['mesodynamic']
            self.meso_static[i] = context[i]['mesostatic']

        # Change all nan's to 0            REMOVE!
        self.highres_dynamic = np.nan_to_num(self.highres_dynamic, nan = 0.0)
        self.highres_static = np.nan_to_num(self.highres_static, nan = 0.0)
        self.meso_dynamic = np.nan_to_num(self.meso_dynamic, nan = 0.0)
        self.meso_static = np.nan_to_num(self.meso_static, nan = 0.0)
 
        ''' Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        self.highres_dynamic = torch.Tensor(self.highres_dynamic).permute(0, 3, 1, 2, 4)
 
    def __len__(self):
        return self.highres_dynamic.shape[0]
 
    def __getitem__(self, index):
        return self.highres_dynamic[index], self.highres_static[index], self.meso_dynamic[index], \
               self.meso_static[index]

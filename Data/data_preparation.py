import earthnet as en
import numpy as np
import torch
import os
from os.path import isfile, join
 
def prepare_data(training_samples, ms_cut, train_dir, test_dir):
    
    train_files = []
    for path, subdirs, files in os.walk(os.getcwd() + train_dir):
        for name in files:
            # Ignore any licence, progress, etc. files
            if '.npz' in name:
                train_files.append(join(path, name))

    test_context_files = []
    test_target_files = []
    for path, subdirs, files in os.walk(os.getcwd() + test_dir):
        for name in files:
            if '.npz' in name:
                full_name = join(path, name)
                if 'context' in full_name:
                    test_context_files.append(full_name)
                else:
                    test_target_files.append(full_name)

    # Sort file names just in case (so we glue together the right context & target)
    test_context_files.sort()
    test_target_files.sort()

    train_files = train_files[:min([training_samples, len(train_files)])]


    train = Earthnet_Dataset(train_files, ms_cut)
    test = Earthnet_Dataset(test_context_files, ms_cut, test_target_files)
    return train, test
 
    train_data = []
    test_context_data = []
    test_target_data = []

    # limit number of training samples
    for i in range(min(len(train_files), training_samples)):
        train_data.append(np.load(train_files[i], allow_pickle=True))
 
    for i in range(len(test_context_files)):
        test_context_data.append(np.load(test_context_files[i], allow_pickle=True))
        test_target_data.append(np.load(test_context_files[i], allow_pickle=True))
    
    train = Earthnet_Dataset(train_data, ms_cut)
    test = Earthnet_Dataset(test_context_data, ms_cut, test_target_data)
    return train, test

class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, context_file_paths, ms_cut, target_file_paths = None):
        '''
            context_file_paths: list of paths of all the context files
            ms_cut: indxs for relevant mesoscale data
            target_file_paths: list of paths of all the context files
        '''
        '''
            The EarthNet dataset combines the different components of a earchnet data cube.

            highres dynamic (hrs) - leave as is as is (for test samples we glue together the contex & target data)
            highres static        - replicate accross all 30 timepoints & add to hrs as an extra channel
            mesoscale dynamic     - cut out the 2x2 center section which overlaps with the actual data cube
                                  - replicate the 4 values to 128 x 128 hrs dimensions & add to hrs as further channels
                                  - Note: ms dynamic data is daily, but we need just 1 value for every 5 day interval
                                       | Precipitation: take the mean
                                       | Sea pressure: take the mean
                                       | Mean temp: take the mean
                                       | Min temp: take the min
                                       | Max temp: take the max
            mesoscale static      - we don't use this data, the relationships are too complex for the model to learn
        '''

        self.context_paths = context_file_paths
        self.target_paths = None
        if target_file_paths is not None:
            self.traget_paths = target_file_paths


        
 
    def __len__(self):
        return len(self.context_paths)
 
    def __getitem__(self, index):
        # load the item from data
        context = np.load(self.context_paths[index], allow_pickle=True)

        if self.target_paths is not None:
            target = np.load(self.target_paths[index], allow_pickle=True)
            highres_dynamic = np.nan_to_num(np.append(context['highresdynamic'], target['highresdynamic'],axis=-1), nan = 0.0)
        else:
            highres_dynamic = np.nan_to_num(context['highresdynamic'], nan = 0.0) 

        highres_static = np.nan_to_num(context['highresstatic'], nan = 0.0)
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)
        meso_static = np.nan_to_num(context['mesostatic'], nan = 0.0)
 
        ''' Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        highres_dynamic = torch.Tensor(highres_dynamic).permute(2, 0, 1, 3)

        # Add up the total number of channels
        channels = context['highresdynamic'].shape[2] + context['highresstatic'].shape[2] + context['mesodynamic'].shape[2]


        return highres_dynamic, highres_static, meso_dynamic, meso_static

        hrs_shape = list(context['highresdynamic'].shape)
        # If target data is given separately add context + target dimensions
        '''
        if target is not None:
            hrs_shape[-1] += target[0]['highresdynamic'].shape[-1]
        '''
        self.highres_dynamic = np.empty((tuple([samples] + hrs_shape)))
        
        self.all = np.empty((tuple([samples] + hrs_shape[0:2] + [channels] + [hrs_shape[3]])))

        self.highres_static = np.empty((tuple([samples] + list(context[0]['highresstatic'].shape))))
        # For mesoscale data we only use the area overlapping the datacube
        self.meso_dynamic = np.empty((tuple([samples] + [ms_cut[1] - ms_cut[0], ms_cut[1] - ms_cut[0]] + list(context[0]['mesodynamic'].shape[2:]))))
        self.meso_static = np.empty((tuple([samples] + [ms_cut[1] - ms_cut[0], ms_cut[1] - ms_cut[0]] + list(context[0]['mesostatic'].shape[2:]))))

        for i in range(samples):
            # For test samples glue together context & target
            if target is not None:
                self.highres_dynamic[i] = np.append(context[i]['highresdynamic'], target[i]['highresdynamic'],axis=-1)
            else:
                self.highres_dynamic[i] = context[i]['highresdynamic']
            self.highres_static[i] = context[i]['highresstatic']
            # For mesoscale data cut out overlapping section of interest
            self.meso_dynamic[i] = context[i]['mesodynamic'][ms_cut[0]:ms_cut[1],ms_cut[0]:ms_cut[1],:,:]
            self.meso_static[i] = context[i]['mesostatic'][ms_cut[0]:ms_cut[1],ms_cut[0]:ms_cut[1],:]


            # self.all[i] = np.append(context[i]['highresdynamic'], context[i]['highresstatic'],axis=-1)

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

        return self.highres_dynamic[index], self.highres_static[index], self.meso_dynamic[index], \
               self.meso_static[index]

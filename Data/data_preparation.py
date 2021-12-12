import earthnet as en
import numpy as np
import torch
import os
from os.path import isfile, join
import random
import math

from torch._C import device
 
def prepare_train_data(ms_cut, data_dir, device, training_samples = None, val_1_samples = None, val_2_samples = None, undersample = False):
    
    if training_samples is not None:
        train_files = []
        if not undersample:
            for path, subdirs, files in os.walk(os.getcwd() + data_dir):    
                for name in files:
                    # Ignore any licence, progress, etc. files
                    if '.npz' in name:
                        train_files.append(join(path, name))
            
            train_files = train_files[:min([training_samples+val_1_samples+val_2_samples, len(train_files)])]
            train_set = random.sample(train_files, training_samples)
            val_set = [x for x in train_files if x not in train_set]
            val_1_set = random.sample(val_set, val_1_samples)
            val_2_set = [x for x in val_set if x not in val_1_set]
            return Earthnet_Dataset(train_set, ms_cut, device=device), \
                Earthnet_Dataset(val_1_set, ms_cut, device=device), \
                Earthnet_Dataset(val_2_set, ms_cut, device=device)
        
        else: # Sample a subset of cube in each tile
            sampling_factor = 5
            for dir in os.scandir(os.getcwd() + data_dir):
                tile_files = []
                for path, subdirs, files in os.walk(dir.path):
                    for name in files:
                        if '.npz' in name:
                            tile_files.append(join(path, name))
                tile_files = random.sample(tile_files, math.ceil(len(tile_files)/sampling_factor))
                train_files = train_files + tile_files
        
            train_files = train_files[:min([training_samples+val_1_samples+val_2_samples, len(train_files)])]
            train_set = random.sample(train_files, math.ceil(training_samples/sampling_factor))
            val_set = [x for x in train_files if x not in train_set]
            val_1_set = random.sample(val_set, math.ceil(val_1_samples/sampling_factor))
            val_2_set = [x for x in val_set if x not in val_1_set]
            return Earthnet_Dataset(train_set, ms_cut, device=device), \
                Earthnet_Dataset(val_1_set, ms_cut, device=device), \
                Earthnet_Dataset(val_2_set, ms_cut, device=device)

def prepare_test_data(ms_cut, data_dir, device):
    
    test_context_files = []
    test_target_files = []
    for path, subdirs, files in os.walk(os.getcwd() + data_dir):
        for name in files:
            if '.npz' in name:
                full_name = join(path, name)
                if 'context' in full_name:
                    test_context_files.append(full_name)
                elif 'target' in full_name:
                    test_target_files.append(full_name)

    # Sort file names just in case (so we glue together the right context & target)
    test_context_files.sort()
    test_target_files.sort()

    return Earthnet_Test_Dataset(test_context_files, test_target_files, ms_cut=ms_cut, device=device)

class Earthnet_Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, context_paths, target_paths, ms_cut, device) -> None:
        self.device = device
        self.context_paths = context_paths
        self.target_paths = target_paths
        self.ms_cut = ms_cut
        assert len(self.context_paths) == len(self.target_paths)
        super().__init__()

    def __len__(self):
        return len(self.context_paths)

    def __getitem__(self, index):
        # Load the item from data
        context = np.load(self.context_paths[index], allow_pickle=True)

        # For test samples glue together context & target
        target = np.load(self.target_paths[index], allow_pickle=True)
        highres_dynamic = np.nan_to_num(np.append(context['highresdynamic'], target['highresdynamic'],axis=-1), nan = 0.0)

        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=highres_dynamic.shape[-1], axis=-1)
        # For mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]

        # Stick all data together
        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = process_md(meso_dynamic, tuple([all_data.shape[0],
                                                            all_data.shape[1],
                                                            meso_dynamic.shape[2],
                                                            all_data.shape[3]]))
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).to(self.device).permute(2, 0, 1, 3)
        
        return all_data

class Earthnet_Context_Dataset(torch.utils.data.Dataset):
    def __init__(self, context_paths, ms_cut, device) -> None:
        self.device = device
        self.context_paths = context_paths
        self.ms_cut = ms_cut
        super().__init__()

    def __len__(self):
        return len(self.context_paths)

    def __getitem__(self, index):
        # Load the item from data
        context = np.load(self.context_paths[index], allow_pickle=True)
        highres_dynamic = np.nan_to_num(context['highresdynamic'], nan = 0.0)
        if (highres_dynamic.shape[2] != 5):
            highres_dynamic = np.append(np.append(highres_dynamic[:,:,0:4,:], highres_dynamic[:,:,6:7,:], axis=2), highres_dynamic[:,:,4:6,:], axis=2)
            # Ignore Cloud mask and ESA scene Classification channels
            highres_dynamic = highres_dynamic[:,:,0:5,:]

        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=highres_dynamic.shape[-1], axis=-1)
        # For mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]

        # Stick all data together
        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = process_md(meso_dynamic, tuple([all_data.shape[0],
                                                            all_data.shape[1],
                                                            meso_dynamic.shape[2],
                                                            all_data.shape[3]]))
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).to(self.device).permute(2, 0, 1, 3)
        all_data = all_data[:,:,:,:10] # Take only context data

        return all_data        

class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, ms_cut, device):
        '''
            context_file_paths: list of paths of all the context files
            ms_cut: indxs for relevant mesoscale data
            target_file_paths: list of paths of all the context files

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
        self.device = device
        self.paths = paths
        self.ms_cut = ms_cut
        
    def __getstate__(self):
        return { 
            "device": self.device.__str__(), 
            "paths": self.paths, 
            "ms_cut": self.ms_cut
        }
        
    def __setstate__(self, d):
        self.device = torch.device(d["device"])
        self.paths = d["paths"]
        self.ms_cut = d["ms_cut"]

    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, index):
        # Load the item from data
        context = np.load(self.paths[index], allow_pickle=True)
        
        highres_dynamic = np.nan_to_num(context['highresdynamic'], nan = 0.0)
        # Make data quality mask the 5th channel
        highres_dynamic = np.append(np.append(highres_dynamic[:,:,0:4,:], highres_dynamic[:,:,6:7,:], axis=2), highres_dynamic[:,:,4:6,:], axis=2)
        # Ignore Cloud mask and ESA scene Classification channels
        highres_dynamic = highres_dynamic[:,:,0:5,:]

        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=highres_dynamic.shape[-1], axis=-1)
        # For mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]

        # Stick all data together
        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = process_md(meso_dynamic, tuple([all_data.shape[0],
                                                            all_data.shape[1],
                                                            meso_dynamic.shape[2],
                                                            all_data.shape[3]]))
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).to(self.device).permute(2, 0, 1, 3)
        
        return all_data
        
def process_md(md, target_shape):
    '''
        Channels: Precipitation (RR), Sea pressure (PP), Mean temperature (TG), Minimum temperature (TN), Maximum temperature (TX)
    '''
    interval = round(md.shape[3] / target_shape[3])

    md_new = np.empty((tuple([md.shape[0], md.shape[1], md.shape[2], target_shape[3]])))
    # Make avg, min, max vals over 5 day intervals
    for i in range(target_shape[3]):
        days = [d for d in range(i*interval, i*interval + interval)]
        for j in range(md.shape[0]):
            for k in range(md.shape[1]):
                md_new[j,k,0,i] = np.mean(md[j,k,0,days])   # mean precipitation
                md_new[j,k,1,i] = np.mean(md[j,k,1,days])   # mean pressure
                md_new[j,k,2,i] = np.mean(md[j,k,2,days])   # mean temp
                md_new[j,k,3,i] = np.min(md[j,k,3,days])    # min temp
                md_new[j,k,4,i] = np.max(md[j,k,4,days])    # max temp

    # Move weather data 1 image forward
    # => the nth image is predicted based on the (n-1)th image and nth weather data
    # the last weather inputed will hence be a null prediction (this should never be used by the model!)
    null_weather = md_new[:,:,:,-1:] * 0
    md_new = np.append(md_new[:,:,:,1:], null_weather, axis=-1)

    # Reshape to 128 x 128
    md_reshaped = np.empty((tuple([target_shape[0], target_shape[1], md.shape[2], md_new.shape[3]])))
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            row = round(i//(target_shape[0]/md.shape[0]))
            col = round(j//(target_shape[1]/md.shape[1]))
            md_reshaped[i,j,:,:] = md_new[row, col,:,:]

    return md_reshaped

class Earthnet_NDVI_Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, ms_cut, device):
        
        '''
            The NDVI Dataset will only consider heavilly vegetated areas of the training dataset
            (since we have access to the ESA Scene Classification)
            We will discard any cubes with a low percetage of vegetated pixels
        '''

        self.device = device
        self.paths = paths
        self.ms_cut = ms_cut
        
    def __getstate__(self):
        return { 
            "device": self.device.__str__(), 
            "paths": self.paths, 
            "ms_cut": self.ms_cut
        }
        
    def __setstate__(self, d):
        self.device = torch.device(d["device"])
        self.paths = d["paths"]
        self.ms_cut = d["ms_cut"]

    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, index):
        # Load the item from data
        context = np.load(self.paths[index], allow_pickle=True)
        
        hrd = np.nan_to_num(context['highresdynamic'], nan = 0.0)
        ndvi = ((hrd[:, :, 3, :] - hrd[ :, :, 2, :]) / (
                    hrd[ :, :, 3, :] + hrd[ :, :, 2, :] + 1e-6))[:, :, np.newaxis, :]
        highres_dynamic = np.append(ndvi, hrd[:,:,6:7,:], axis=2)
        # Maybe we also want the sencor cloud mask in here?
        
        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=highres_dynamic.shape[-1], axis=-1)
        # For mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]

        # Stick all data together
        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = process_md(meso_dynamic, tuple([all_data.shape[0],
                                                            all_data.shape[1],
                                                            meso_dynamic.shape[2],
                                                            all_data.shape[3]]))
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).to(self.device).permute(2, 0, 1, 3)
        
        return all_data 

def process_md(md, target_shape):
    '''
        Channels: Precipitation (RR), Sea pressure (PP), Mean temperature (TG), Minimum temperature (TN), Maximum temperature (TX)
    '''
    interval = round(md.shape[3] / target_shape[3])

    md_new = np.empty((tuple([md.shape[0], md.shape[1], md.shape[2], target_shape[3]])))
    # Make avg, min, max vals over 5 day intervals
    for i in range(target_shape[3]):
        days = [d for d in range(i*interval, i*interval + interval)]
        for j in range(md.shape[0]):
            for k in range(md.shape[1]):
                md_new[j,k,0,i] = np.mean(md[j,k,0,days])   # mean precipitation
                md_new[j,k,1,i] = np.mean(md[j,k,1,days])   # mean pressure
                md_new[j,k,2,i] = np.mean(md[j,k,2,days])   # mean temp
                md_new[j,k,3,i] = np.min(md[j,k,3,days])    # min temp
                md_new[j,k,4,i] = np.max(md[j,k,4,days])    # max temp

    # Move weather data 1 image forward
    # => the nth image is predicted based on the (n-1)th image and nth weather data
    # the last weather inputed will hence be a null prediction (this should never be used by the model!)
    null_weather = md_new[:,:,:,-1:] * 0
    md_new = np.append(md_new[:,:,:,1:], null_weather, axis=-1)

    # Reshape to 128 x 128
    md_reshaped = np.empty((tuple([target_shape[0], target_shape[1], md.shape[2], md_new.shape[3]])))
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            row = round(i//(target_shape[0]/md.shape[0]))
            col = round(j//(target_shape[1]/md.shape[1]))
            md_reshaped[i,j,:,:] = md_new[row, col,:,:]

    return md_reshaped
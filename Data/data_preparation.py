import earthnet as en
import numpy as np
import torch
import os
from os.path import isfile, join

 
def prepare_data(training_samples, ms_cut, train_dir, test_dir, device):
    
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

    # Save paths to test set for ENS calculation
    '''
    with open(os.getcwd() + test_dir + '/target_files' + timestamp + '.txt', 'w') as filehandle:
        for item in test_target_files:
            filehandle.write('%s\n' % item)
    '''

    train_files = train_files[:min([training_samples, len(train_files)])]

    train = Earthnet_Dataset(train_files, ms_cut, device=device)
    test = Earthnet_Dataset(test_context_files, ms_cut, target_file_paths = test_target_files, device=device)
    return train, test

class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, context_file_paths, ms_cut, device, target_file_paths = None):
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
        self.context_paths = context_file_paths
        self.ms_cut = ms_cut
        self.target_paths = target_file_paths

    def process_md(self, md, target_shape):
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
 
    def __len__(self):
        return len(self.context_paths)
 
    def __getitem__(self, index):
        # Load the item from data
        context = np.load(self.context_paths[index], allow_pickle=True)

        # For test samples glue together context & target
        if self.target_paths is not None:
            target = np.load(self.target_paths[index], allow_pickle=True)
            highres_dynamic = np.nan_to_num(np.append(context['highresdynamic'], target['highresdynamic'],axis=-1), nan = 0.0)
        else:
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

        meso_dynamic = self.process_md(meso_dynamic, tuple([all_data.shape[0],
                                                            all_data.shape[1],
                                                            meso_dynamic.shape[2],
                                                            all_data.shape[3]]))
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data, device=self.device).permute(2, 0, 1, 3)
        
        if self.target_paths is not None:
            return all_data, [self.target_paths[index]]
        else:
            return all_data, ["no target"]


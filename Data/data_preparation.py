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

class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, context_file_paths, ms_cut, target_file_paths = None):
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

        self.context_paths = context_file_paths
        self.ms_cut = ms_cut
        self.target_paths = target_file_paths

    def process_md(self, md, target_shape):
        '''
            Channels: Precipitation (RR), Sea pressure (PP), Mean temperature (TG), Minimum temperature (TN), Maximum temperature (TX)
        '''
        interval = round(md.shape[3] / target_shape[3])

        md_new = np.empty((tuple([md.shape[0]] + [md.shape[1]] + [md.shape[2]] + [30])))
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

        # Reshape to 128 x 128
        md_reshaped = np.empty((tuple([target_shape[0]] + [target_shape[1]] + [md.shape[2]] + [md_new.shape[3]])))
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
        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=30, axis=-1)
        # For mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]
        meso_static = np.nan_to_num(context['mesostatic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:]

        # Add up the total number of channels
        channels = highres_dynamic.shape[2] + highres_static.shape[2] + meso_dynamic.shape[2]

        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = self.process_md(meso_dynamic, tuple([all_data.shape[0]] + [all_data.shape[1]] + 
                                            [meso_dynamic.shape[2]] + [all_data.shape[3]]))
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        highres_dynamic = torch.Tensor(highres_dynamic).permute(2, 0, 1, 3)
        all_data = torch.Tensor(all_data).permute(2, 0, 1, 3)
        

        return highres_dynamic, highres_static, meso_dynamic, meso_static, all_data

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




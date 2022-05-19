import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import os
from os.path import join
import pickle

class Earthnet_Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, ms_cut, fake_weather = False, all_weather = False):
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
        self.paths = paths
        self.ms_cut = ms_cut
        self.fake_weather = False
        
    def __setstate__(self, d):
        self.paths = d["paths"]
        self.ms_cut = d["ms_cut"]

    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, index):
        # load the item from data
        context = np.load(self.paths[index], allow_pickle=True)
        
        highres_dynamic = np.nan_to_num(context['highresdynamic'], nan = 0.0)
        # make data quality mask the 5th channel
        highres_dynamic = np.append(np.append(highres_dynamic[:,:,0:4,:], highres_dynamic[:,:,6:7,:], axis=2), highres_dynamic[:,:,4:6,:], axis=2)
        # ignore Cloud mask and ESA scene Classification channels
        highres_dynamic = highres_dynamic[:,:,0:5,:]

        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=highres_dynamic.shape[-1], axis=-1)
        # for mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]

        # stick all data together
        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = process_md(meso_dynamic, tuple([all_data.shape[0],
                                                       all_data.shape[1],
                                                       meso_dynamic.shape[2],
                                                       all_data.shape[3]]))
        if self.fake_weather:
            meso_dynamic = 0*meso_dynamic
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).permute(2, 0, 1, 3)
        
        return all_data

class Earthnet_Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, context_paths, target_paths, ms_cut, fake_weather = False, all_weather = False) -> None:
        self.context_paths = context_paths
        self.target_paths = target_paths
        self.ms_cut = ms_cut
        self.fake_weather = fake_weather
        assert len(self.context_paths) == len(self.target_paths)
        super().__init__()

    def __len__(self):
        return len(self.context_paths)

    def __getitem__(self, index):
        # load the item from data
        context = np.load(self.context_paths[index], allow_pickle=True)

        # for test samples glue together context & target
        target = np.load(self.target_paths[index], allow_pickle=True)
        highres_dynamic = np.nan_to_num(np.append(context['highresdynamic'], target['highresdynamic'],axis=-1), nan = 0.0)

        highres_static = np.repeat(np.expand_dims(np.nan_to_num(context['highresstatic'], nan = 0.0), axis=-1), repeats=highres_dynamic.shape[-1], axis=-1)
        # for mesoscale data cut out overlapping section of interest
        meso_dynamic = np.nan_to_num(context['mesodynamic'], nan = 0.0)[self.ms_cut[0]:self.ms_cut[1],self.ms_cut[0]:self.ms_cut[1],:,:]

        # stick all data together
        all_data = np.append(highres_dynamic, highres_static,axis=-2)

        meso_dynamic = process_md(meso_dynamic, tuple([all_data.shape[0],
                                                       all_data.shape[1],
                                                       meso_dynamic.shape[2],
                                                       all_data.shape[3]]))
        if self.fake_weather:
            meso_dynamic = 0*meso_dynamic 
        all_data = np.append(all_data, meso_dynamic, axis=-2)
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).permute(2, 0, 1, 3)
        
        return all_data

def process_md(md, target_shape):
    '''
        Channels: Precipitation (RR), Sea pressure (PP), Mean temperature (TG), Minimum temperature (TN), Maximum temperature (TX)
    '''
    interval = round(md.shape[3] / target_shape[3])

    md_new = np.empty((tuple([md.shape[0], md.shape[1], md.shape[2], target_shape[3]])))
    # make avg, min, max vals over 5 day intervals
    for i in range(target_shape[3]):
        days = [d for d in range(i*interval, i*interval + interval)]
        for j in range(md.shape[0]):
            for k in range(md.shape[1]):
                md_new[j,k,0,i] = np.mean(md[j,k,0,days])   # mean precipitation
                md_new[j,k,1,i] = np.mean(md[j,k,1,days])   # mean pressure
                md_new[j,k,2,i] = np.mean(md[j,k,2,days])   # mean temp
                md_new[j,k,3,i] = np.min(md[j,k,3,days])    # min temp
                md_new[j,k,4,i] = np.max(md[j,k,4,days])    # max temp

    # move weather data 1 image forward
    # => the nth image is predicted based on the (n-1)th image and nth weather data
    # the last weather inputed will hence be a null prediction (this should never be used by the model!)
    null_weather = md_new[:,:,:,-1:] * 0
    md_new = np.append(md_new[:,:,:,1:], null_weather, axis=-1)

    # reshape to 128 x 128
    md_reshaped = np.empty((tuple([target_shape[0], target_shape[1], md.shape[2], md_new.shape[3]])))
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            row = round(i//(target_shape[0]/md.shape[0]))
            col = round(j//(target_shape[1]/md.shape[1]))
            md_reshaped[i,j,:,:] = md_new[row, col,:,:]

    return md_reshaped

class Earth_net_DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./", 
                 train_batch_size = 4,
                 val_batch_size = 4,
                 test_batch_size = 4,
                 mesoscale_cut = [39, 41],
                 test_set = 'val_2',
                 fake_weather = False,
                 all_weather = False):
        """
        This is wrapper for all of our datasets. It preprocesses the data into a format that can be fed into our model.

        Parameters:
            data_dir: Location of pickle file with paths to the train/validation/test datapoints
            mesoscale_cut: The coordinates of the mesoscale data channels that overlap with our satellite image data
            train_batch_size: Number of data points in a single train batch
            val_batch_size: Number of data points in a single validation batch
            test_batch_size: Number of data points in a single test batch
            test_set: Which test set to use
            fake_weather: Whether to use 'fake_weather'
            train_batch_size: Number of data points in a single train batch
        """
        super().__init__()
        self.data_dir = data_dir
        self.mesoscale_cut = mesoscale_cut
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.test_set = test_set
        self.fake_weather = fake_weather
        self.all_weather = all_weather

        with open(join(os.getcwd(), self.data_dir, "train_data_paths.pkl"),'rb') as f:
            self.training_path_list = pickle.load(f)
        with open(join(os.getcwd(), self.data_dir, "val_1_data_paths.pkl"),'rb') as f:
            self.val_1_path_list = pickle.load(f)
        if test_set == 'val_2':
            with open(join(os.getcwd(), self.data_dir, "val_2_data_paths.pkl"),'rb') as f:
                self.val_2_path_list = pickle.load(f)
        else: 
            with open(join(os.getcwd(), self.data_dir, self.test_set+"_context_data_paths.pkl"),'rb') as f:
                self.test_context_path_list = pickle.load(f)
            with open(join(os.getcwd(), self.data_dir, self.test_set+"_target_data_paths.pkl"),'rb') as f:
                self.test_target_path_list = pickle.load(f)

        if 'train_targ_data_paths.pkl' in os.listdir(join(os.getcwd(), self.data_dir)):
            with open(join(os.getcwd(), self.data_dir, "train_targ_data_paths.pkl"),'rb') as f:
                self.training_targ_path_list = pickle.load(f)
            with open(join(os.getcwd(), self.data_dir, "val_1_targ_data_paths.pkl"),'rb') as f:
                self.val_1_targ_path_list = pickle.load(f)
            # in the seasonal training data case there is no 'extra' test set
            with open(join(os.getcwd(), self.data_dir, "val_2_targ_data_paths.pkl"),'rb') as f:
                self.val_2_targ_path_list = pickle.load(f)

    def setup(self, stage):
        # assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            # training
            self.training_data = Earthnet_Dataset(self.training_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)
            # validation
            self.val_1_data = Earthnet_Dataset(self.val_1_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)

            if hasattr(self, 'training_targ_path_list'):
                self.training_data = Earthnet_Test_Dataset(self.training_path_list, self.training_targ_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)
                self.val_1_data = Earthnet_Test_Dataset(self.val_1_path_list, self.val_1_targ_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)

        # assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            if self.test_set == 'val_2':
                self.val_2_data = Earthnet_Dataset(self.val_2_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)
            else:
                self.test_data = Earthnet_Test_Dataset(self.test_context_path_list, self.test_target_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)
            
            if hasattr(self, 'training_targ_path_list'):
                self.val_2_data = Earthnet_Test_Dataset(self.val_2_path_list, self.val_2_targ_path_list, self.mesoscale_cut, fake_weather=self.fake_weather, all_weather = self.all_weather)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_1_data, batch_size=self.val_batch_size)

    def test_dataloader(self):
        if self.test_set == 'val_2':
            return DataLoader(self.val_2_data, batch_size=self.test_batch_size)
        else:
            return DataLoader(self.test_data, batch_size=self.test_batch_size)

    # store pickle files together with model for reproducibility
    def serialize_datasets(self, directory):
        with open(join(directory, "train_data_paths.pkl"), "wb") as fp:
            pickle.dump(self.training_path_list, fp)
        with open(join(directory, "val_1_data_paths.pkl"), "wb") as fp:
            pickle.dump(self.val_1_path_list, fp)
        with open(join(directory, "val_2_data_paths.pkl"), "wb") as fp:
            pickle.dump(self.val_2_path_list, fp)

from logging import error
import numpy as np
from numpy.ma.core import masked_where
import torch

def mean_cube(cube, mask_channel = False): #dumb one
    # cube is the input cube (note that the time is always the last coordinate)
    # cannels is the list of channels we compute the avarage on
    # mask_channel whether we include the data quality mask (if we include the mask channel it should be the last one)
    '''
    cube dimensions will be:
        b, c, w, h, t
    '''
    if not mask_channel:
        return cube.mean(axis=-1)
    else:
        channels = cube.shape[1]
        mask = np.repeat(cube[:,-1:,:,:,:], channels-1, axis=1)
        cube_to_mask = cube[:,:-1,:,:,:]
        # Mask which data is cloudy and shouldn't be used for averaging
        masked_cube = np.ma.masked_where(mask, cube_to_mask)
        avg_cube = masked_cube.mean(axis=-1)
        return torch.tensor(avg_cube).float()
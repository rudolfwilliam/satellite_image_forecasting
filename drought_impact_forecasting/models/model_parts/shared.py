from logging import error
import numpy as np
from numpy.ma.core import masked_where
import torch
import earthnet as en

def mean_cube(cube, mask_channel = False): #dumb one
    # cube is the input cube (note that the time is always the last coordinate)
    # cannels is the list of channels we compute the avarage on
    # mask_channel whether we include the data quality mask (if we include the mask channel it should be the last one)
    '''
    cube dimensions will be:
        b, c, w, h, t
    '''
    if not mask_channel:
        return torch.mean(cube, dim=-1)
    else:
        channels = cube.shape[1]
        # Mask which data is cloudy and shouldn't be used for averaging
        mask = torch.repeat_interleave(1-cube[:,-1:,:,:,:], channels-1, axis=1)

        masked_cube = mask * cube[:,:-1,:,:,:]
        avg_cube = torch.mean(masked_cube, dim=-1)
        return avg_cube

def last_cube(cube, mask_channel = 4):
    #note that cube can either be a torch tensor or a numpy arrat
    new_cube = mean_cube(cube[:, 0:4, :, :, :])

    # for each pixel, find the last good quality data point
    # if no data point has good quality return the mean
    for i in range(cube.shape[2]):
        for j in range(cube.shape[3]):
            for k in reversed(range(cube.shape[mask_channel])):
                if cube[0,mask_channel,i,j,k] == 0:
                    new_cube[0, :4, i, j] = cube[0,:4,i,j,k]
                    break
    return new_cube

def mean_prediction(cube, mask_channel = False, timepoints = 20):
    # compute the mean image and make a prediction cube of the correct length
    avg_cube = np.array(mean_cube(cube[:, 0:5, :, :, :], True)).transpose(2,3,1,0)
    return np.repeat(avg_cube, timepoints, axis=-1)

def last_prediction(cube, mask_channel = 4, timepoints = 20):
    # find the last cloud free context image and return it as a constant prediction

    new_cube = last_cube(cube, mask_channel)
    return np.repeat(new_cube.permute(2,3,1,0), timepoints, axis=-1)

def get_ENS(target, preds):
    # Calculate the ENS score of each prediction in preds
    scores = []
    for pred in preds:
        output = en.parallel_score.CubeCalculator.get_scores({"pred_filepath": pred, "targ_filepath": target})
        if output['MAD'] == 0 or output['OLS'] == 0 or output['EMD'] == 0 or output['SSIM'] == 0:
            scores.append(0)
        else:
            denom = 1/output['MAD'] + 1/output['OLS'] + 1/output['EMD'] + 1/output['SSIM']
            scores.append(4/denom)    
    return scores
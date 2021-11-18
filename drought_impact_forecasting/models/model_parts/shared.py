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
    # note that cube can either be a torch tensor or a numpy array
    new_cube = mean_cube(cube[:, 0:4, :, :, :])

    # for each pixel, find the last good quality data point
    # if no data point has good quality return the mean
    for c in range(cube.shape[0]):
        for i in range(cube.shape[2]):
            for j in range(cube.shape[3]):
                for k in reversed(range(cube.shape[mask_channel])):
                    if cube[c,mask_channel,i,j,k] == 0:
                        new_cube[c, :4, i, j] = cube[c,:4,i,j,k]
                        break
    return new_cube

def last_frame(cube, mask_channel = 4):
    # Note that by default the last channel will be the mask
    T = cube.shape[-1]
    # 1 = good quality, 0 = bad quality (in the flipped version)   
    mask = 1 - cube[:, mask_channel, :, :, T - 1] 
    new_cube = cube[:, :4, :, :, T - 1] * mask

    t = T - 1
    while (torch.max(mask) > 0 and t >= 0):
        mask = (1 - mask) * (1 - cube[:, mask_channel, :, :, t])
        new_cube += cube[:, :4, :, :, t] * mask
        t -= 1
    return new_cube

def mean_prediction(cube, mask_channel = False, timepoints = 20):
    # compute the mean image and make a prediction cube of the correct length
    avg_cube = np.array(mean_cube(cube[:, 0:5, :, :, :], True)).transpose(2,3,1,0)
    return np.repeat(avg_cube, timepoints, axis=-1)

def last_prediction(cube, mask_channel = 4, timepoints = 20):
    # find the last cloud free context image and return it as a constant prediction

    new_cube = last_frame(cube, mask_channel)
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

def ENS(target, prediction):
    mask = 1 - np.repeat(target[:,:,4:,:], 4, axis=2)
    target = target[:,:,:4,:]
    ndvi_target = ((target[:,:,3,...] - target[:,:,2,...])/(target[:,:,3,...] + target[:,:,2,...] + 1e-6))[:,:,np.newaxis,...]
    ndvi_prediction = ((prediction[:,:,3,...] - prediction[:,:,2,...])/(prediction[:,:,3,...] + prediction[:,:,2,...] + 1e-6))[:,:,np.newaxis,...]
    ndvi_mask = mask[:,:,:1,:]
    # Partial score
    mad, _ = en.parallel_score.CubeCalculator.MAD(target, prediction, mask)
    ssim, _ = en.parallel_score.CubeCalculator.SSIM(target, prediction, mask)
    ols, _ = en.parallel_score.CubeCalculator.OLS(ndvi_target, ndvi_prediction, ndvi_mask)
    emd, _ = en.parallel_score.CubeCalculator.EMD(ndvi_target, ndvi_prediction, ndvi_mask) # this is the slow one
    if mad == 0 or ssim == 0 or ols == 0 or emd == 0:
        return 0
    else:
        return 4/(1/mad + 1/ssim + 1/ols + 1/emd)
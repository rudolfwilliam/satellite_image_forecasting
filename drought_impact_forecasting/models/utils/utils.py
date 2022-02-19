import numpy as np
import torch
import earthnet as en

def mean_cube(cube, mask_channel=False):  # dumb one
    # cube is the input cube (note that the time is always the last coordinate)
    # cannels is the list of channels we compute the avarage on
    # mask_channel whether we include the data quality mask (if we include the mask channel it should be the last one)
    '''
    cube dimensions are:
        b, c, w, h, t
    '''
    if not mask_channel:
        return torch.mean(cube, dim=-1)
    else:
        channels = cube.shape[1]
        # mask which data is cloudy and shouldn't be used for averaging
        mask = torch.repeat_interleave(1 - cube[:, -1:, :, :, :], channels - 1, axis=1)

        masked_cube = mask * cube[:, :-1, :, :, :] 
        avg_cube = torch.sum(masked_cube, dim=-1) / torch.sum(mask, dim = -1)
        return torch.nan_to_num(avg_cube, nan = 0)

def last_cube(cube, mask_channel=4):
    # note that cube can either be a torch tensor or a numpy array
    new_cube = mean_cube(cube[:, 0:4, :, :, :])

    # for each pixel, find the last good quality data point
    # if no data point has good quality return the mean
    for c in range(cube.shape[0]):
        for i in range(cube.shape[2]):
            for j in range(cube.shape[3]):
                for k in reversed(range(cube.shape[mask_channel])):
                    if cube[c, mask_channel, i, j, k] == 0:
                        new_cube[c, :4, i, j] = cube[c, :4, i, j, k]
                        break
    return new_cube

def last_frame(cube, mask_channel=4):
    # note that by default the last channel will be the mask
    T = cube.shape[-1]
    # 1 = good quality, 0 = bad quality (in the flipped version)
    mask = 1 - cube[:, mask_channel:mask_channel + 1, :, :, T - 1]
    missing = cube[:, mask_channel:mask_channel + 1, :, :, T - 1]  # 1 = is missing, 0 = is already assigned
    new_cube = cube[:, :-1, :, :, T - 1] * mask

    t = T - 1
    while (torch.min(mask) == 0 and t >= 0):
        mask = missing * (1 - cube[:, mask_channel:mask_channel + 1, :, :, t])
        new_cube += cube[:, :-1, :, :, t] * mask
        missing = missing * (1 - mask)
        t -= 1
    return new_cube
    
def zeros(cube, mask_channel = 4):
    return cube[:, :-1, :, :, 0]*0

def mean_prediction(cube, mask_channel=True, timepoints=20):
    # compute the mean image and make a prediction cube of the correct length
    avg_cube = mean_cube(cube[:, 0:5, :, :, :], mask_channel).permute(2, 3, 1, 0)
    return avg_cube.repeat(1,1,1,1,timepoints)
    
def last_prediction(cube, mask_channel=4, timepoints=20):
    # find the last cloud free context image and return it as a constant prediction
    new_cube = last_frame(cube, mask_channel).permute(2, 3, 1, 0)
    return new_cube.repeat(1,1,1,1,timepoints)

def get_ENS(target, preds):
    # calculate the ENS score of each prediction in preds
    scores = []
    for pred in preds:
        output = en.parallel_score.CubeCalculator.get_scores({"pred_filepath": pred, "targ_filepath": target})
        if output['MAD'] == 0 or output['OLS'] == 0 or output['EMD'] == 0 or output['SSIM'] == 0:
            scores.append(0)
        else:
            denom = 1 / output['MAD'] + 1 / output['OLS'] + 1 / output['EMD'] + 1 / output['SSIM']
            scores.append(4 / denom)
    return scores

def ENS(target: torch.Tensor, prediction: torch.Tensor):
    '''
        target of size (b, w, h, c, t)
            b = batch_size (>0)
            c = channels (=5)
            w = width (=128)
            h = height (=128)
            t = time (20/40/140)
        target of prediction (b, w, h, c, t)
            b = batch_size (>0)
            c = channels (=4) no mask
            w = width (=128)
            h = height (=128)
            t = time (20/40/140)
    '''
    # numpy conversion
    target = np.array(target.cpu()).transpose(0, 2, 3, 1, 4)
    prediction = np.array(prediction.cpu()).transpose(0, 2, 3, 1, 4)

    # mask
    mask = 1 - np.repeat(target[:, :, :, 4:, :], 4, axis=3)
    target = target[:, :, :, :4, :]

    # NDVI
    ndvi_target = ((target[:, :, :, 3, :] - target[:, :, :, 2, :]) / (
                target[:, :, :, 3, :] + target[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
    ndvi_prediction = ((prediction[:, :, :, 3, :] - prediction[:, :, :, 2, :]) / (
                prediction[:, :, :, 3, :] + prediction[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
    ndvi_mask = mask[:, :, :, 0, :][:, :, :, np.newaxis, :]

    # floor and ceiling
    prediction[prediction < 0] = 0
    prediction[prediction > 1] = 1

    target[np.isnan(target)] = 0
    target[target > 1] = 1
    target[target < 0] = 0

    partial_score = np.zeros((target.shape[0], 5))
    score = np.zeros(target.shape[0])

    # partial score computation
    for i in range(target.shape[0]):
        partial_score[i, 1], _ = en.parallel_score.CubeCalculator.MAD(prediction[i], target[i], mask[i])
        partial_score[i, 2], _ = en.parallel_score.CubeCalculator.SSIM(prediction[i], target[i], mask[i])
        partial_score[i, 3], _ = en.parallel_score.CubeCalculator.OLS(ndvi_prediction[i], ndvi_target[i], ndvi_mask[i])
        partial_score[i, 4], _ = en.parallel_score.CubeCalculator.EMD(ndvi_prediction[i], ndvi_target[i], ndvi_mask[i])
        if np.min(partial_score[i, 1:]) == 0:
            score[i] = partial_score[i, 0] = 0
        else:
            score[i] = partial_score[i, 0] = 4 / (
                        1 / partial_score[i, 1] + 1 / partial_score[i, 2] + 1 / partial_score[i, 3] + 1 / partial_score[i, 4])

    return score, partial_score
    # score is a np array with all the scores
    # partial scores is np array with 5 columns, ENS mad ssim ols emd, in this order (one row per elem in batch)

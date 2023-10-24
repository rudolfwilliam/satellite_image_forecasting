import torch
from torch import nn
import numpy as np
import earthnet as en

def get_loss_from_name(loss_name):
    if loss_name == "l2":
        return Cube_loss(nn.MSELoss())
    elif loss_name == "l1":
        return Cube_loss(nn.L1Loss())
    elif loss_name == "Huber":
        return Cube_loss(nn.HuberLoss())
    elif loss_name == "ENS":
        return ENS_loss()
    elif loss_name == "ENS_by_section":
        return ENS_by_section_loss()
    elif loss_name == "NDVI":
        return NDVI_loss()

# simple L2 loss on the RGBI channels, mostly used for training
class Cube_loss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.l = loss
    
    def forward(self, labels: torch.Tensor, prediction: torch.Tensor):
        # only compute loss on non-cloudy pixels
        mask = 1 - labels[:, 4:5] # [b, 1, h, w, t]
        mask = mask.repeat(1, 4, 1, 1, 1)
        masked_prediction = prediction * mask
        masked_labels = labels[:, :4] * mask
        return self.l(masked_prediction, masked_labels)

# loss using the EarthNet challenge ENS score
class ENS_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels: torch.Tensor, prediction: torch.Tensor):
        '''
            size of labels (b, w, h, c, t)
                b = batch_size (>0)
                c = channels (=5)
                w = width (=128)
                h = height (=128)
                t = time (20/40/140)
            size of prediction (b, w, h, c, t)
                b = batch_size (>0)
                c = channels (=4) no mask
                w = width (=128)
                h = height (=128)
                t = time (20/40/140)
        '''
        # numpy conversion
        labels = np.array(labels.cpu()).transpose(0, 2, 3, 1, 4)
        prediction = np.array(prediction.cpu()).transpose(0, 2, 3, 1, 4)

        # mask
        mask = 1 - np.repeat(labels[:, :, :, 4:, :], 4, axis=3)
        labels = labels[:, :, :, :4, :]

        # NDVI
        ndvi_labels = ((labels[:, :, :, 3, :] - labels[:, :, :, 2, :]) / (
                    labels[:, :, :, 3, :] + labels[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
        ndvi_prediction = ((prediction[:, :, :, 3, :] - prediction[:, :, :, 2, :]) / (
                    prediction[:, :, :, 3, :] + prediction[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
        ndvi_mask = mask[:, :, :, 0, :][:, :, :, np.newaxis, :]

        # floor and ceiling
        prediction[prediction < 0] = 0
        prediction[prediction > 1] = 1

        labels[np.isnan(labels)] = 0
        labels[labels > 1] = 1
        labels[labels < 0] = 0

        partial_score = np.zeros((labels.shape[0], 5))
        score = np.zeros(labels.shape[0])
        # partial score computation
        for i in range(labels.shape[0]):
            partial_score[i, 1], _ = en.parallel_score.CubeCalculator.MAD(prediction[i], labels[i], mask[i])
            partial_score[i, 2], _ = en.parallel_score.CubeCalculator.SSIM(prediction[i], labels[i], mask[i])
            partial_score[i, 3], _ = en.parallel_score.CubeCalculator.OLS(ndvi_prediction[i], ndvi_labels[i], ndvi_mask[i])
            partial_score[i, 4], _ = en.parallel_score.CubeCalculator.EMD(ndvi_prediction[i], ndvi_labels[i], ndvi_mask[i])
            if np.min(partial_score[i, 1:]) == 0:
                score[i] = partial_score[i, 0] = 0
            else:
                score[i] = partial_score[i, 0] = 4 / (
                            1 / partial_score[i, 1] + 1 / partial_score[i, 2] + 1 / partial_score[i, 3] + 1 / partial_score[i, 4])
        
        return score, partial_score
        # score is a np array with all the scores
        # partial scores is np array with 5 columns, ENS mad ssim ols emd, in this order (one row per elem in batch)

# losses by 20 time steps (used for the seasonal test set)
class ENS_by_section_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels: torch.Tensor, prediction: torch.Tensor):
        '''
            size of labels (b, w, h, c, t)
                b = batch_size (>0)
                c = channels (=5)
                w = width (=128)
                h = height (=128)
                t = time (20/40/140)
            size of prediction (b, w, h, c, t)
                b = batch_size (>0)
                c = channels (=4) no mask
                w = width (=128)
                h = height (=128)
                t = time (20/40/140)
        '''
        # sections are: 0-19, 20-30, 40-59, 60-79, 80-99, 100-119, 120-139
        num_sections = 7

        # numpy conversion
        labels = np.array(labels.cpu()).transpose(0, 2, 3, 1, 4)
        prediction = np.array(prediction.cpu()).transpose(0, 2, 3, 1, 4)

        # mask
        mask = 1 - np.repeat(labels[:, :, :, 4:, :], 4, axis=3)
        labels = labels[:, :, :, :4, :]

        # NDVI
        ndvi_labels = ((labels[:, :, :, 3, :] - labels[:, :, :, 2, :]) / (
                    labels[:, :, :, 3, :] + labels[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
        ndvi_prediction = ((prediction[:, :, :, 3, :] - prediction[:, :, :, 2, :]) / (
                    prediction[:, :, :, 3, :] + prediction[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
        ndvi_mask = mask[:, :, :, 0, :][:, :, :, np.newaxis, :]

        # floor and ceiling
        prediction[prediction < 0] = 0
        prediction[prediction > 1] = 1

        labels[np.isnan(labels)] = 0
        labels[labels > 1] = 1
        labels[labels < 0] = 0

        #partial_score = np.zeros((labels.shape[0], 5))
        partial_score_by_section = np.zeros((labels.shape[0], num_sections, 5))
        score = np.zeros(labels.shape[0])

        # TODO: see of this loop could be somehow optimized
        # partial score computation
        for i in range(labels.shape[0]):
            for j in range(num_sections):
                interval_start = j*20
                interval_end = j*20+20

                cur_preds = prediction[..., interval_start:interval_end]
                cur_labels = labels[..., interval_start:interval_end]
                cur_ndvi_prediction = ndvi_prediction[..., interval_start:interval_end]
                cur_ndvi_labels = ndvi_labels[..., interval_start:interval_end]
                cur_mask = mask[..., interval_start:interval_end]
                cur_ndvi_mask = ndvi_mask[..., interval_start:interval_end]

                # Calculate the scores per section
                partial_score_by_section[i, j, 1], _ = en.parallel_score.CubeCalculator.MAD(cur_preds[i], cur_labels[i], cur_mask[i])
                partial_score_by_section[i, j, 2], _ = en.parallel_score.CubeCalculator.SSIM(cur_preds[i], cur_labels[i], cur_mask[i])
                partial_score_by_section[i, j, 3], _ = en.parallel_score.CubeCalculator.OLS(cur_ndvi_prediction[i], cur_ndvi_labels[i], cur_ndvi_mask[i])
                partial_score_by_section[i, j, 4], _ = en.parallel_score.CubeCalculator.EMD(cur_ndvi_prediction[i], cur_ndvi_labels[i], cur_ndvi_mask[i])
                if np.min(partial_score_by_section[i, j, 1:]) == 0:
                    partial_score_by_section[i, j, 0] = 0
                else:
                    partial_score_by_section[i, j, 0] = 4 / (
                                1 / partial_score_by_section[i, j, 1] + 1 / partial_score_by_section[i, j, 2] + 1 / partial_score_by_section[i, j, 3] + 1 / partial_score_by_section[i, j, 4])


            '''partial_score[i, 1], _ = en.parallel_score.CubeCalculator.MAD(prediction[i], labels[i], mask[i])
            partial_score[i, 2], _ = en.parallel_score.CubeCalculator.SSIM(prediction[i], labels[i], mask[i])
            partial_score[i, 3], _ = en.parallel_score.CubeCalculator.OLS(ndvi_prediction[i], ndvi_labels[i], ndvi_mask[i])
            partial_score[i, 4], _ = en.parallel_score.CubeCalculator.EMD(ndvi_prediction[i], ndvi_labels[i], ndvi_mask[i])
            if np.min(partial_score[i, 1:]) == 0:
                score[i] = partial_score[i, 0] = 0
            else:
                score[i] = partial_score[i, 0] = 4 / (
                            1 / partial_score[i, 1] + 1 / partial_score[i, 2] + 1 / partial_score[i, 3] + 1 / partial_score[i, 4])'''
        
        # Flatten out partial_score_by_section into a 2D array
        PBBS_flattened = partial_score_by_section.reshape(labels.shape[0], 35)

        return score, PBBS_flattened
        # score is a np array with all the scores
        # partial scores is np array with 5 columns, ENS mad ssim ols emd, in this order (one row per elem in batch)

# NDVI l2 loss on non-cloudy pixels weighted by the proportion of valid pixels
class NDVI_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, labels: torch.Tensor, prediction: torch.Tensor):
        # only compute loss on non-cloudy pixels
        # numpy conversion
        labels = labels.permute(0, 2, 3, 1, 4)
        prediction = prediction.permute(0, 2, 3, 1, 4)

        # mask
        ndvi_mask = labels[:, :, :, 4:, :]
        labels = labels[:, :, :, :4, :]

        # NDVI
        ndvi_labels = ((labels[:, :, :, 3, :] - labels[:, :, :, 2, :]) / (
                    labels[:, :, :, 3, :] + labels[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]
        ndvi_prediction = ((prediction[:, :, :, 3, :] - prediction[:, :, :, 2, :]) / (
                    prediction[:, :, :, 3, :] + prediction[:, :, :, 2, :] + 1e-6))[:, :, :, np.newaxis, :]

        # floor and ceiling
        prediction[prediction < 0] = 0
        prediction[prediction > 1] = 1

        score = np.zeros((labels.shape[0], 6))
        weight = np.zeros(ndvi_labels.shape[0])
        l2_loss = nn.MSELoss()

        for i in range(ndvi_labels.shape[0]):
            # mask which data is cloudy and shouldn't be used for calculating the score
            masked_ndvi_labels = torch.mul(ndvi_labels[i], ndvi_mask[i])
            masked_ndvi_prediction = torch.mul(ndvi_prediction[i], ndvi_mask[i])
            score[i, 0] = score[i, 1] = score[i, 2] = score[i, 3] = score[i, 4] = l2_loss(masked_ndvi_prediction, masked_ndvi_labels)

            # the loss should carry a weight corresponding to the number of valid pixels
            weight[i] = 1 - torch.sum(ndvi_mask[i]) / torch.numel(ndvi_mask[i])
            score[i,5] = weight[i]
        return weight, score

def cloud_mask_loss(y_preds, y_truth, cloud_mask):
    l2_loss = nn.MSELoss()

    mask = torch.repeat_interleave(1-cloud_mask, 4, axis=1)
    # mask which data is cloudy and shouldn't be used for averaging
    masked_y_pred = torch.mul(y_preds, mask)
    masked_y_truth = torch.mul(y_truth, mask)
    return l2_loss(masked_y_pred, masked_y_truth)

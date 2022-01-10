import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import earthnet as en
from tqdm import tqdm

def get_loss_from_name(loss_name):
    if loss_name == "l2":
        return Cube_loss(nn.MSELoss())
    elif loss_name == "l1":
        return Cube_loss(nn.L1Loss())
    elif loss_name == "Huber":
        return Cube_loss(nn.HuberLoss())
    elif loss_name == "ENS":
        return ENS_loss()

class Cube_loss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.l = loss
    def forward(self, labels: torch.Tensor, prediction: torch.Tensor):
        mask = 1 - labels[:,4:5] # [b, 1, h, w, t]
        mask = mask.repeat(1, 4, 1, 1, 1)
        masked_prediction = prediction * mask
        masked_labels = labels[:, :4] * mask
        return self.l(masked_prediction, masked_labels)

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

class L2_disc_cube_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, labels):
        return 1

# Linearly anneal the LK loss weight for certain epochs
def kl_weight(epoch, finalWeight, annealStart, annealEnd):
    if epoch <= annealStart:
        return 0
    elif epoch > annealEnd:
        return finalWeight
    else:
        return finalWeight * (epoch - annealStart)/(annealEnd - annealStart)

# Add up (and weight) the different loss components
def base_line_total_loss(y_preds, batch_y, epoch, lambda1, lambda_kl_factor, annealStart, annealEnd):
    l1_criterion = nn.L1Loss() 
    kl_criterion = nn.KLDivLoss()
    GAN_criterion = nn.CrossEntropyLoss()
    VAE_GAN_Criterion = nn.CrossEntropyLoss()

    lambda_kl_final = lambda1 * lambda_kl_factor
    curlambda_kl = kl_weight(epoch, lambda_kl_final, annealStart, annealEnd)

    L1 = l1_criterion(y_preds, batch_y).mul(lambda1)
    L_KL = kl_criterion(y_preds, batch_y).mul(curlambda_kl)
    L_GAN = GAN_criterion(y_preds, batch_y)
    # Later on these will have to come from the 2nd discriminator
    # I suggest we start by making it work with just the GAN discriminator for now
    L_VAE_GAN = VAE_GAN_Criterion(y_preds, batch_y)

    # TODO: Add variants (GAN only, VAE only)
    loss_total = L1.add(L_KL).add(L_GAN).add(L_VAE_GAN)
    return loss_total

def cloud_mask_loss(y_preds, y_truth, cloud_mask):
    l2_loss = nn.MSELoss()

    mask = torch.repeat_interleave(1-cloud_mask, 4, axis=1)
    # Mask which data is cloudy and shouldn't be used for averaging
    masked_y_pred = torch.mul(y_preds, mask)
    masked_y_truth = torch.mul(y_truth, mask)
    return l2_loss(masked_y_pred,masked_y_truth)

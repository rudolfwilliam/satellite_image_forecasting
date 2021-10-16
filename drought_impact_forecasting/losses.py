import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


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

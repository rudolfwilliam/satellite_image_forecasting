import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

lambda1 = 100
# Note: In the paper they use 2 diferent ways to set the lambdaKLFactor
lambdaKLFactor = 0.001
annealStart = 5 # 50000 in SAVP
annealEnd = 10 # 100000 in SAVP

# Linearly anneal the LK loss weight for certain epochs
def getKLWeight(epoch, finalWeight):
    if epoch <= annealStart:
        return 0
    elif epoch >= annealEnd:
        return finalWeight
    else:
        return finalWeight * (epoch - annealStart)/(annealEnd - annealStart)

# Add up (and weight) the different loss components
def getTotalLoss(y_preds, batch_y, epoch):
    L1Criterion = nn.L1Loss() 
    KLCriterion = nn.KLDivLoss()
    GANCriterion = nn.CrossEntropyLoss()
    VAEGANCriterion = nn.CrossEntropyLoss()

    lambdaKLFinal = lambda1 * lambdaKLFactor
    curlambdaKL = getKLWeight(epoch, lambdaKLFinal)

    L1 = L1Criterion(y_preds, batch_y).mul(lambda1)
    L_KL = KLCriterion(y_preds, batch_y).mul(curlambdaKL)
    L_GAN = GANCriterion(y_preds, batch_y)
    # Later on these will have to come from the 2nd discriminator
    # I suggest we start by making it work with just the GAN discriminator for now
    L_VAE_GAN = VAEGANCriterion(y_preds, batch_y)

    # TODO: Add variants (GAN only, VAE only)
    LossTotal = L1.add(L_KL).add(L_GAN).add(L_VAE_GAN)
    return LossTotal        

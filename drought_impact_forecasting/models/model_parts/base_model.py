import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import math
import matplotlib.pyplot as plt

epochs = 30  # 300000 in real SAVP
startLearnRate = 0.05
decayPoint = 10 # 100000 in real SAVP
batchSize = 32

# The encoder uses LeakyReLU!
class Encoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class Decoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

# The 2 Discrininators share same architecture, but not weights
# 3D convolutional neural net
class Discriminator_VAE(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class Discriminator_GAN(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        # 1st shot at net architecture
        conv_layer1 = self._conv_layer_set(3, 32)
        conv_layer2 = self._conv_layer_set(32, 64)

        fc1 = nn.Linear(64*28*28*28, 2) 
        fc2 = nn.Linear(1404928, classes) 
        relu = nn.LeakyReLU()
        
        self.layers = [conv_layer1, conv_layer2, fc1, fc2, relu]
        self.net = torch.nn.Sequential(*self.layers)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        return F.softmax(self.net(x))

# Linearly decay the learning rate for the last set of training epochs
def getLearningRate(epoch):
    decayPoint = 2*epochs/3
    if epoch < decayPoint:
        return startLearnRate
    else: startLearnRate * ((epoch-decayPoint)/(epochs-decayPoint))

# Linearly anneal the LK loss weight for certain epochs
def getKLWeight(epoch, finalWeight):
    annealStart = 5 # 50000 in SAVP
    annealEnd = 10 # 100000 in SAVP

    if epoch <= annealStart:
        return 0
    elif epoch >= annealEnd:
        return finalWeight
    else:
        return finalWeight * (epoch - annealStart)/(annealEnd - annealStart)

# TODO: Not yet sure how to implement this one (not in Pytorch losses)
def getVAEGANLoss(pred, target):
    return 0

def train_network(model, optimizer, train_loader, noEpochs, pbar_update_interval=100):

    L1Criterion = nn.L1Loss() 
    KLCriterion = nn.KLDivLoss()
    GANCriterion = nn.CrossEntropyLoss()

    lambda1 = 100
    # Note: In the paper they use 2 diferent ways to set lambdaKL
    lambdaKLFinal = lambda1 * 0.001
    L1Loss = nn.L1Loss()

    for i in noEpochs:
        print("Iteration: " + str(i))
        curlambdaKL = getKLWeight(i, lambdaKLFinal)
        curLearningRate = getLearningRate()

        for k, (batch_x, batch_y) in enumerate(train_loader):
            model.zero_grad()
            y_preds = model(batch_x)
            
            L1 = L1Criterion(y_preds, batch_y).mul(lambda1)
            L_KL = KLCriterion(y_preds, batch_y).mul(curlambdaKL)
            L_GAN = GANCriterion(y_preds, batch_y)
            L_VAE_GAN = getVAEGANLoss(y_preds, batch_y)

            LossTotal = L1.add(L_KL).add(L_GAN).add(L_VAE_GAN)
            
            # TODO: Add variants (GAN only, VAE only)

            LossTotal.backward()
            optimizer.step()



def main():
    


    optimizer = torch.optim.Adam(model.parameters(), lr=startLearnRate)

    train_network()


if __name__=="__main__":
    main()
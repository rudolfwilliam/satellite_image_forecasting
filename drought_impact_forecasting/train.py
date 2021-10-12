import torch
from torch import nn

import os
import sys
from pathlib import Path
import json

from .models.model_parts.base_model import *
from .losses import *

# Later these will be loaded from config/SAVP.json
epochs = 30  # 300000 in real SAVP
startLearnRate = 0.05
decayPoint = 10 # 100000 in real SAVP
batchSize = 32
classes = 10
decayPoint = 20

# Here I would start simply with 1 datacube (see Data folder) just rgb values
# to see if the network is learning properly
def load_train_data():
    return None

# Linearly decay the learning rate for the last set of training epochs
def getLearningRate(epoch):
    if epoch < decayPoint:
        return startLearnRate
    else: startLearnRate * ((epoch-decayPoint)/(epochs-decayPoint))



def train_network(model, optimizer, train_loader, noEpochs):

    L1Criterion = nn.L1Loss() 
    KLCriterion = nn.KLDivLoss()
    GANCriterion = nn.CrossEntropyLoss()


    for i in noEpochs:
        print("Iteration: " + str(i))
        
        curLearningRate = getLearningRate(i)
        for g in optimizer.param_groups:
            g['lr'] = curLearningRate

        for k, (batch_x, batch_y) in enumerate(train_loader):
            model.zero_grad()
            y_preds = model(batch_x)
            
            
            LossTotal = getTotalLoss(y_preds, batch_y, i)

            LossTotal.backward()
            optimizer.step()


def main():
    
    # Start just with a GAN, once it works we can make a VAE-GAN
    model = Discriminator_GAN(classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=startLearnRate)

    dataset_train = load_train_data()
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize,
                                               shuffle=True, drop_last=True)

    train_network(model, optimizer, train_loader, num_epochs=epochs)


if __name__=="__main__":
    main()
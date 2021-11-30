import sys
import os
import numpy as np
import random
from shutil import copy2

sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl

from config.config import command_line_parser
from Data.data_preparation import prepare_train_data, prepare_test_data
from drought_impact_forecasting.models.utils.utils import last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS
from datetime import datetime

def main():

    # This should run the 2 'dumb' baseline models: copy-pasteing the 10th image & copy pasteting the mean of img 0-9

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") # timestamp unique to this training instance
    print("Timestamp of the instance: " + timestamp)

    args, cfg = command_line_parser(mode='train')
    print(args)

    instance_folder = os.getcwd() + "/wandb/model_" + timestamp
    os.mkdir(instance_folder)
    copy2(os.getcwd() + "/config/" + args.model_name + ".json", instance_folder + "/" + cfg["model"]["baseline"] + ".json")

    #GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("GPU count: {0}".format(gpu_count))

    training_data, val_1_data, val_2_data = prepare_train_data(cfg["data"]["mesoscale_cut"],
                                                         cfg["data"]["train_dir"],
                                                         device = device,
                                                         training_samples=cfg["training"]["training_samples"],
                                                         val_1_samples=cfg["training"]["val_1_samples"],
                                                         val_2_samples=cfg["training"]["val_2_samples"])
   
    test_data = prepare_test_data(cfg["data"]["mesoscale_cut"], 
                             cfg["data"]["test_dir"],
                             device = device)

    with open(instance_folder + "/scores.csv", 'w') as filehandle:
                filehandle.write("mad, ssim, ols, emd, score\n")

    for i in range(len(val_2_data.paths)):
        all_data = val_2_data.__getitem__(i)

        T = all_data.size()[3]

        t0 = round(all_data.shape[-1]/3) #t0 is the length of the context part

        # For last/mean baseline we don't need weather
        context = all_data[:5, :, :, :t0].unsqueeze(0) # b, c, h, w, t
        target = all_data[:5, :, :, t0:].unsqueeze(0) # b, c, h, w, t

        baseline = eval(cfg["model"]["baseline"] + '(context, 4)')
        preds = baseline.unsqueeze(-1).repeat(1,1,1,1,T-t0)

        score, part_scores = ENS(prediction = preds, target = target)

        with open(instance_folder + "/scores.csv", 'a') as filehandle:
            filehandle.write(str(part_scores[0,1]) + "," +str(part_scores[0,2]) + "," + str(part_scores[0,3]) + "," + str(part_scores[0,4])+ "," + str(score[0]) + '\n')

if __name__ == "__main__":
    main()


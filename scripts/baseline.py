import sys
import os
from shutil import copy2

sys.path.append(os.getcwd())

import torch

from config.config import train_line_parser
from Data.data_preparation import prepare_train_data, prepare_test_data, Earth_net_DataModule
from drought_impact_forecasting.models.utils.utils import last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS
from datetime import datetime

def main():

    # This should run the 2 'dumb' baseline models: copy-pasteing the 10th image & copy pasteting the mean of img 0-9

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") # timestamp unique to this training instance
    print("Timestamp of the instance: " + timestamp)

    cfg = train_line_parser()

    instance_folder = os.getcwd() + "/wandb/model_" + timestamp
    os.mkdir(instance_folder)
    copy2(os.getcwd() + "/config/Training.json", instance_folder + "/" + cfg["model"]["baseline"] + ".json")

    #GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("GPU count: {0}".format(gpu_count))

    '''training_data, val_1_data, val_2_data = prepare_train_data(cfg["data"]["mesoscale_cut"],
                                                         cfg["data"]["train_dir"],
                                                         device=device,
                                                         training_samples=cfg["training"]["training_samples"],
                                                         val_1_samples=cfg["training"]["val_1_samples"],
                                                         val_2_samples=cfg["training"]["val_2_samples"])

    with open(os.path.join(instance_folder, "train_data_paths.pkl"), "wb") as fp:
        pickle.dump(training_data.paths, fp)

    val_2_data = training_data
    test_data = prepare_test_data(cfg["data"]["mesoscale_cut"], 
                             cfg["data"]["test_dir"],
                             device = device)'''
    
    # Always use same val_2 data from Data folder
    EN_dataset = Earth_net_DataModule(data_dir = cfg['data']['pickle_dir'],
                                     test_set = 'iid_test_split',
                                     train_batch_size = 1,
                                     val_batch_size = 1,
                                     test_batch_size = 1,
                                     mesoscale_cut = [39,41])
    EN_dataset.setup(stage = 'test')

    test_DL = EN_dataset.test_dataloader()
    print("Using " + str(test_DL.__len__()) + " samples")
    for i in range(test_DL.__len__()):
        all_data = test_DL.dataset.__getitem__(i)

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


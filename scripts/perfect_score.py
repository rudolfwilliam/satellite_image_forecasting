import argparse
import os
from os.path import join
from re import S
import numpy as np
import pickle
import torch
import sys

sys.path.append(os.getcwd())

from drought_impact_forecasting.models.utils.utils import ENS

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('-tf', '--test_folder', type=str, help='folder where the test data is', default=None)
args = parser.parse_args()

with open(join(os.getcwd(), args.test_folder, "test_target_data_paths.pkl"),'rb') as f:
    paths = pickle.load(f)

# calculate the ENS score when comparing the target datacubes against themselves
# note that due to the cloud masks this will not always be 1!
for path in paths:
    context = np.load(path, allow_pickle=True)
    highres_dynamic = np.nan_to_num(context['highresdynamic'], nan = 0.0)
    highres_dynamic = np.append(np.append(highres_dynamic[:,:,0:4,:], highres_dynamic[:,:,6:7,:], axis=2), highres_dynamic[:,:,4:6,:], axis=2)
    
    # bcwht

    highres_dynamic = torch.Tensor(highres_dynamic[np.newaxis,:,:,0:5,:]).permute(0,3,1,2,4)
    _, score = ENS(highres_dynamic,highres_dynamic[:,:4,:,:,:])
    with open("perfect_score.csv", "a") as f:
        f.write(str(score[0,1]) + "," + str(score[0,2]) + "," + str(score[0,3])+ "," + str(score[0,4]) + ","+ str(score[0,0]) + '\n')

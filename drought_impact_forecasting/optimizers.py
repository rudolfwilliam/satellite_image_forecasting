import torch
from torch import nn
import torch.optim as optim

def get_opt_from_name(opt_name, params, lr=0.01):
    if opt_name == "adam":
        return optim.Adam(params, lr=lr)
    elif opt_name == "adamW":
        return optim.AdamW(params, lr=lr)
    elif opt_name == "SGD":
        return optim.SGD(params, lr=lr)
    elif opt_name == "ASGD":
        return optim.ASGD(params, lr=lr)
    else:
        raise ValueError("You have specified an invalid optimizer.")
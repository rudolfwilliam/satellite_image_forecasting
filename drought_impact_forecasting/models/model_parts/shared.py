from logging import error
import numpy as np
import torch

def mean_cube(cube, mask_channel = False): #dumb one
    # cube is the input cube (note that the time is always the last coordinate)
    # cannels is the list of channels we compute the avarage on
    # mask_channel is the channel of the mask channel
    # is_firts_index_batch = True implies that the first index is just the b, we don't avarage over it
    '''
    cube dimensions will be:
        b, c, w, h, t (c, w, h, t if is_firts_index_batch is False)
    '''
    if not mask_channel:
        return cube.mean(axis=-1)
    else:
        error("TODO: mask_channel not yet implemented")
    # TODO: mask_channel not yet implemented
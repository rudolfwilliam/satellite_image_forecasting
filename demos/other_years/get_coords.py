import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import earthnet as en
from os import walk
import os
from my_coord import get_coords_from_cube as my_coord


import os
from pyproj import Transformer
from earthnet.coords_dict import COORDS

NEWCOORD = {
    "32UPC" : {
        "MinLon" : 10.4361,
        "MaxLat" : 52.3414,
        'EPSG': 32632
    },
    "32UMC" : {
        "MinLon" : 7.5315,
        "MaxLat" : 52.3504,
        'EPSG': 32632
    },
    "32UNC" : {
        "MinLon" : 8.9997,
        "MaxLat" : 52.314,
        'EPSG': 32632
    },
    "32UQC" : { #This has been aligned!
        "MinLon" : 11.9352,
        "MaxLat" : 52.315,
        'EPSG': 32632
    }
}

def get_coords(cubename: str, return_meso: bool = False):
    cubetile,_, _,  hr_y_min, hr_y_max,hr_x_min, hr_x_max, meso_x_min, meso_x_max, meso_y_min, meso_y_max = os.path.splitext(cubename)[0].split("_")

    tile = NEWCOORD[cubetile]

    transformer = Transformer.from_crs(tile["EPSG"], 4326, always_xy = True)

    tile_x_min, tile_y_max = transformer.transform(tile["MinLon"],tile["MaxLat"], direction = "INVERSE")

    a = 20
    tile_x_min = tile_x_min

    cube_x_min = tile_x_min + a * float(hr_x_min)
    cube_x_max = tile_x_min + a * float(hr_x_max)
    cube_y_min = tile_y_max - a * float(hr_y_min)
    cube_y_max = tile_y_max - a * float(hr_y_max)
    
    cube_lon_min, cube_lat_min = transformer.transform(cube_x_min, cube_y_max)
    cube_lon_max, cube_lat_max = transformer.transform(cube_x_max, cube_y_min)

    #print("{0}, {1}".format(cube_lat_min,cube_lon_min))
    cube_lat_min
    
    return cube_lon_min, cube_lat_min, cube_lon_max, cube_lat_max
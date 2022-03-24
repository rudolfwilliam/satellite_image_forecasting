import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import earthnet as en
from os import walk
import os


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
locs = {
    "32UQC_2018-01-28_2018-11-23_5305_5433_4665_4793_82_162_72_152":{
        "lon_max": 13.243973,
        "lon_min": 13.208011,
        "lat_max": 51.321081, 
        "lat_min": 51.297443
    },
    "32UQC_2018-01-28_2018-11-23_5305_5433_4409_4537_82_162_68_148":{
        "lon_max": 13.172049,
        "lon_min": 13.136258,
        "lat_max": 51.323078, 
        "lat_min": 51.299554
    }
}
def get_limited_coords(cubename: str):
    coord = locs[cubename]
    return coord["lon_min"], coord["lat_min"], coord["lon_max"], coord["lat_max"]
    
def get_coords(cubename: str):
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